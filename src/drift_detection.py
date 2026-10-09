"""
drift_detection.py — Evidently drift job for the Rakuten pipeline.

Compares what the model is being asked to classify now against the data it
was trained on, writes an HTML report, records the summary in the database
(so the API can expose it to Prometheus) and logs everything to MLflow.
Optionally asks the API to retrain when drift is found.

Run it on a schedule (cron, or an Airflow DAG):
    python src/drift_detection.py --current-limit 500

What is actually compared
-------------------------
reference : rows from the `products` table (the historical training data),
            with their TRUE prdtypecode.
current   : rows from the `predictions` table (recent live traffic), with
            the model's PREDICTED prdtypecode.

So two different things are measured at once, and it is worth being precise
about which is which:

  * text_length / word_count  -> input drift. Same quantity on both sides;
    a shift means the products being sent to the API no longer look like
    the ones it was trained on.
  * prdtypecode               -> prediction drift. True labels on one side,
    predicted ones on the other. A shift means the model's output mix has
    moved away from the historical class balance, which can mean the traffic
    changed OR that the model degraded. It is a symptom, not a diagnosis.

Ground-truth drift cannot be measured here: live predictions have no labels.
That would need a feedback loop, which is out of scope for this project and
is called out as such in the README.

This module deliberately avoids importing TensorFlow. It runs in its own
environment (and its own container), because Evidently pulls in a large
dependency tree that cannot coexist with the pinned TensorFlow stack.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = os.getenv("RAKUTEN_REPORTS_DIR", "reports/drift")
NUMERICAL_FEATURES = ["text_length", "word_count"]
CATEGORICAL_FEATURES = ["prdtypecode"]

# Share of drifted columns above which the dataset counts as drifted enough
# to act on. Evidently reports its own boolean too; this threshold is what
# the retraining decision uses, and it is explicit so it can be defended.
DRIFT_ACTION_THRESHOLD = float(os.getenv("RAKUTEN_DRIFT_THRESHOLD", "0.5"))


def build_reference(limit, db_path=None):
    """Historical training data, with the same feature columns as `current`."""
    rows = db.fetch_reference_sample(limit=limit, db_path=db_path)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    features = frame.apply(
        lambda row: db.text_features(row["designation"], row["description"]),
        axis=1,
        result_type="expand",
    )
    frame = pd.concat([frame, features], axis=1)
    frame["prdtypecode"] = frame["prdtypecode"].astype(str)
    return frame[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]


def build_current(limit, db_path=None):
    """Recent live predictions, with the features stored at prediction time."""
    rows = db.fetch_recent_predictions(limit=limit, db_path=db_path)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    # text_length / word_count were computed and stored by the API; recompute
    # only if an older row predates that.
    missing = frame["text_length"].isna()
    if missing.any():
        recomputed = frame.loc[missing].apply(
            lambda row: db.text_features(row["designation"], row["description"]),
            axis=1,
            result_type="expand",
        )
        frame.loc[missing, NUMERICAL_FEATURES] = recomputed[NUMERICAL_FEATURES]

    frame["prdtypecode"] = frame["prdtypecode"].astype(str)
    return frame[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]


def _extract_summary(report_dict):
    """Pull the dataset-level numbers out of Evidently's report.

    Searched by key rather than by position: the order of metrics inside a
    preset is not part of Evidently's public contract.
    """
    for entry in report_dict.get("metrics", []):
        result = entry.get("result", {})
        if "share_of_drifted_columns" in result:
            return {
                "dataset_drift": bool(result.get("dataset_drift", False)),
                "n_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
                "share_drifted_columns": float(
                    result.get("share_of_drifted_columns", 0.0)
                ),
                "n_columns": int(result.get("number_of_columns", 0)),
            }
    raise ValueError("No dataset-level drift metric found in the Evidently report")


def run_drift_check(
    reference_limit=1000,
    current_limit=500,
    db_path=None,
    log_to_mlflow=True,
    reports_dir=REPORTS_DIR,
):
    """Run one drift check end to end. Returns the summary dict."""
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    reference = build_reference(reference_limit, db_path=db_path)
    current = build_current(current_limit, db_path=db_path)

    if reference.empty or current.empty:
        logger.warning(
            "Not enough data to compare (reference=%d, current=%d rows). "
            "Serve some predictions first.",
            len(reference),
            len(current),
        )
        return {"status": "skipped", "reference_rows": len(reference),
                "current_rows": len(current)}

    mapping = ColumnMapping(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target=None,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    summary = _extract_summary(report.as_dict())
    summary.update(
        {
            "status": "ok",
            "reference_rows": len(reference),
            "current_rows": len(current),
            "threshold": DRIFT_ACTION_THRESHOLD,
            "action_required": summary["share_drifted_columns"]
            >= DRIFT_ACTION_THRESHOLD,
        }
    )

    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(reports_dir, f"drift_{stamp}.html")
    report.save_html(report_path)
    logger.info("Drift report written to %s", report_path)

    run_id = _log_to_mlflow(summary, report_path) if log_to_mlflow else None

    db.log_drift_report(
        dataset_drift=summary["dataset_drift"],
        n_drifted_columns=summary["n_drifted_columns"],
        share_drifted_columns=summary["share_drifted_columns"],
        n_reference_rows=summary["reference_rows"],
        n_current_rows=summary["current_rows"],
        mlflow_run_id=run_id,
        report_path=report_path,
        db_path=db_path,
    )

    summary["report_path"] = report_path
    summary["mlflow_run_id"] = run_id
    return summary


def _log_to_mlflow(summary, report_path):
    """Track the report and its metrics. Never fatal."""
    try:
        import mlflow

        import mlflow_utils

        mlflow_utils.setup_experiment(
            os.getenv("MLFLOW_DRIFT_EXPERIMENT", "rakuten-drift")
        )
        with mlflow.start_run() as run:
            mlflow.log_metrics(
                {
                    "dataset_drift": float(summary["dataset_drift"]),
                    "n_drifted_columns": float(summary["n_drifted_columns"]),
                    "share_drifted_columns": summary["share_drifted_columns"],
                    "reference_rows": float(summary["reference_rows"]),
                    "current_rows": float(summary["current_rows"]),
                }
            )
            mlflow.log_artifact(report_path, artifact_path="drift")
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not log the drift report to MLflow: %s", exc)
        return None


def trigger_retraining(api_url, params=None):
    """Ask the API to retrain. Used when drift crosses the threshold.

    Calling the API rather than importing training.py on purpose: this job
    runs in an environment without TensorFlow, and the API already serialises
    training runs behind one lock.
    """
    import requests

    payload = params or {"samples_per_class": 100}
    response = requests.post(f"{api_url.rstrip('/')}/training/", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _parse_args():
    parser = argparse.ArgumentParser(description="Evidently drift check")
    parser.add_argument("--reference-limit", type=int, default=1000)
    parser.add_argument("--current-limit", type=int, default=500)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=REPORTS_DIR)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--retrain-on-drift",
        type=str,
        default=None,
        metavar="API_URL",
        help="Trigger POST <API_URL>/training/ when drift crosses the threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outcome = run_drift_check(
        reference_limit=args.reference_limit,
        current_limit=args.current_limit,
        db_path=args.db_path,
        log_to_mlflow=not args.no_mlflow,
        reports_dir=args.reports_dir,
    )

    if args.retrain_on_drift and outcome.get("action_required"):
        logger.info("Drift above threshold — requesting a retraining run")
        try:
            outcome["retraining"] = trigger_retraining(args.retrain_on_drift)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not trigger retraining: %s", exc)
            outcome["retraining_error"] = str(exc)

    print(json.dumps(outcome, indent=2, default=str))
