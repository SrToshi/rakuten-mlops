"""
rakuten_drift_check — compare live traffic against the training data, and
retrain when it has moved far enough.

    check     POST /run on the drift service — Evidently compares recent
              predictions against a sample of the training data, writes an
              HTML report, logs the metrics to MLflow and stores the summary
    gate      a short circuit: stop here unless the drifted share crossed the
              threshold
    retrain   trigger rakuten_training

The Evidently call runs in the drift service rather than here. That is the
same reason the training DAG calls the API: Evidently pulls Litestar and its
own Uvicorn extras, TensorFlow pins typing_extensions below what MLflow's
server needs, and none of it belongs in an orchestrator's image. Airflow says
WHEN and in WHAT ORDER; the services say HOW.

What drift means here, stated plainly because it is easy to overclaim:

  * text_length and word_count are the same quantity on both sides — a shift
    means the products being sent to the API no longer look like the ones it
    was trained on.
  * prdtypecode compares TRUE labels in the reference against PREDICTED ones
    in the current window. A shift means the output mix moved, which can mean
    the traffic changed or the model degraded. It is a symptom, not a
    diagnosis.

Ground-truth drift is not measurable: live predictions have no labels. That
would need a feedback loop, which this project does not have.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DRIFT_URL = os.getenv("RAKUTEN_DRIFT_URL", "http://drift:8100").rstrip("/")

REFERENCE_LIMIT = int(os.getenv("DRIFT_REFERENCE_LIMIT", "1000"))
CURRENT_LIMIT = int(os.getenv("DRIFT_CURRENT_LIMIT", "500"))

default_args = {
    "owner": "rakuten",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def run_drift_check(**context):
    """Run one Evidently comparison and push its summary to XCom."""
    response = requests.post(
        f"{DRIFT_URL}/run",
        json={
            "reference_limit": REFERENCE_LIMIT,
            "current_limit": CURRENT_LIMIT,
            "log_to_mlflow": True,
        },
        timeout=600,
    )
    response.raise_for_status()
    summary = response.json()

    if summary.get("status") == "skipped":
        # Nothing has been predicted yet. An idle system is not a broken one.
        print(
            f"Skipped: {summary.get('reference_rows')} reference rows, "
            f"{summary.get('current_rows')} current rows"
        )
        return summary

    print(
        f"Dataset drift: {summary['dataset_drift']} · "
        f"{summary['n_drifted_columns']} of the monitored columns drifted "
        f"({summary['share_drifted_columns']:.0%}) · "
        f"compared {summary['current_rows']} recent predictions against "
        f"{summary['reference_rows']} training rows"
    )
    print(f"Report: {summary.get('report_path')}")
    print(f"MLflow run: {summary.get('mlflow_run_id')}")
    return summary


def drift_requires_retraining(**context):
    """Gate. Returning False short-circuits everything downstream.

    The threshold lives in the drift service, which reports `action_required`
    rather than making the DAG re-derive it — one definition, one place.
    """
    summary = context["ti"].xcom_pull(task_ids="run_drift_check")

    if not summary or summary.get("status") != "ok":
        print("No usable drift summary — not retraining")
        return False

    if summary.get("action_required"):
        print(
            f"Drifted share {summary['share_drifted_columns']:.0%} is at or "
            f"above the {summary['threshold']:.0%} threshold — retraining"
        )
        return True

    print(
        f"Drifted share {summary['share_drifted_columns']:.0%} is below the "
        f"{summary['threshold']:.0%} threshold — nothing to do"
    )
    return False


with DAG(
    dag_id="rakuten_drift_check",
    description="Evidently drift check; triggers retraining when it crosses the threshold",
    default_args=default_args,
    start_date=datetime(2026, 10, 1),
    schedule="0 * * * *",  # hourly
    catchup=False,
    max_active_runs=1,
    tags=["rakuten", "monitoring"],
) as dag:

    check = PythonOperator(
        task_id="run_drift_check",
        python_callable=run_drift_check,
    )

    gate = ShortCircuitOperator(
        task_id="drift_above_threshold",
        python_callable=drift_requires_retraining,
    )

    retrain = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="rakuten_training",
        # Do not block this DAG for the hours the training takes: the drift
        # check should keep running on its own schedule meanwhile.
        wait_for_completion=False,
        reset_dag_run=True,
    )

    check >> gate >> retrain
