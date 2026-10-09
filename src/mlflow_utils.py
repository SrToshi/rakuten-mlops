"""
mlflow_utils.py — MLflow helpers for the Rakuten project.

Keeps every MLflow-specific concern in one place so that training.py and
api.py stay readable, and so this logic can be unit-tested without
TensorFlow or the dataset being present.

Responsibilities:
  * resolve the tracking URI (env-driven, so Docker and local agree)
  * register a newly trained model version in the Model Registry
  * compare it against the current champion and tag the winner
  * resolve / download the champion so the API can serve it

Design notes
------------
* "Champion vs challenger" is expressed with model-version TAGS
  (`stage=champion` / `stage=challenger`), which work on every MLflow 2.x
  and 3.x. A registry ALIAS (`@champion`) is also set when the server
  supports it, because aliases are the modern replacement for the
  deprecated Model Stages — but the tags remain the source of truth.
* Every public function degrades gracefully: if the tracking server is
  unreachable, training must still finish and the API must still serve
  from local files. MLflow is observability, not a hard dependency.
"""

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Default experiment / registered-model names. Overridable via env so the
# Docker compose stack and a local run can point at the same server.
DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "rakuten-classification")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "rakuten-fusion")

# The metric that decides which model version is champion. Higher is better.
#
# Scored on the held-out TEST split. It was previously computed on
# validation, which both branches already used for early stopping and which
# the blend search also used — so the deciding number came from rows that had
# influenced fitting. Versions registered before that change carry the old
# metric name and therefore have no comparable value; promote_if_better()
# treats a missing champion metric as "nothing to lose" and promotes, which
# is the right behaviour for the first run under the new definition.
PRIMARY_METRIC = "ensemble_test_weighted_f1"

CHAMPION = "champion"
CHALLENGER = "challenger"


def get_tracking_uri() -> str:
    """Tracking URI, from MLFLOW_TRACKING_URI, else a local file store.

    The file store is the default on purpose. The training environment runs
    `mlflow-skinny` (the client), because the full server package drags in
    SQLAlchemy and Alembic, which demand a typing_extensions newer than the
    one TensorFlow 2.13 pins — the server and the trainer simply cannot share
    one environment. Without SQLAlchemy a `sqlite://` URI cannot be opened,
    while the file store serves both tracking and the Model Registry.

    In Docker this is overridden with the tracking server's HTTP URL, and the
    server container keeps its own SQLite backend.
    """
    return os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def setup_experiment(experiment_name: str = DEFAULT_EXPERIMENT) -> str:
    """Point MLflow at the tracking server and select/create the experiment."""
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(experiment_name)
    return experiment_name


def _version_metric(client: MlflowClient, version, metric_name: str):
    """Read a metric off the run that produced a given model version.

    Returns None when the run is gone or never logged that metric, so the
    caller can treat it as 'no comparable champion'.
    """
    try:
        run = client.get_run(version.run_id)
    except Exception:  # run deleted, or server hiccup
        return None
    return run.data.metrics.get(metric_name)


def get_champion_version(
    model_name: str = REGISTERED_MODEL_NAME,
    client: MlflowClient = None,
):
    """Return the ModelVersion currently tagged `stage=champion`, or None."""
    client = client or MlflowClient(tracking_uri=get_tracking_uri())
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        logger.warning("Could not query the model registry: %s", exc)
        return None

    champions = [v for v in versions if v.tags.get("stage") == CHAMPION]
    if not champions:
        return None
    # Defensive: if several are tagged (manual edits), the highest version wins.
    return max(champions, key=lambda v: int(v.version))


def promote_if_better(
    new_version,
    metric_name: str = PRIMARY_METRIC,
    model_name: str = REGISTERED_MODEL_NAME,
    client: MlflowClient = None,
) -> dict:
    """Compare a freshly registered version with the reigning champion.

    This is the "load the previous version and compare it with the newly
    trained model" step of the roadmap. The new version is promoted only
    when it strictly beats the champion on `metric_name`; otherwise it is
    kept as a challenger and the champion is left untouched.

    Returns a small dict describing what happened, which the training
    endpoint can hand straight back to its caller.
    """
    client = client or MlflowClient(tracking_uri=get_tracking_uri())

    new_metric = _version_metric(client, new_version, metric_name)
    champion = get_champion_version(model_name, client=client)
    champion_metric = (
        _version_metric(client, champion, metric_name) if champion else None
    )

    # First ever model, or the previous champion has no comparable metric:
    # there is nothing to lose by promoting.
    if champion is None or champion_metric is None or new_metric is None:
        promoted = True
    else:
        promoted = new_metric > champion_metric

    if promoted:
        if champion is not None and champion.version != new_version.version:
            client.set_model_version_tag(
                model_name, champion.version, "stage", CHALLENGER
            )
        client.set_model_version_tag(model_name, new_version.version, "stage", CHAMPION)
        _set_alias_if_supported(client, model_name, CHAMPION, new_version.version)
    else:
        client.set_model_version_tag(
            model_name, new_version.version, "stage", CHALLENGER
        )

    # Always record the deciding metric on the version itself, so the
    # registry UI is readable without drilling into each run.
    if new_metric is not None:
        client.set_model_version_tag(
            model_name, new_version.version, metric_name, f"{new_metric:.4f}"
        )

    return {
        "promoted": promoted,
        "new_version": new_version.version,
        "new_metric": new_metric,
        "previous_champion_version": champion.version if champion else None,
        "previous_champion_metric": champion_metric,
        "metric_name": metric_name,
    }


def _set_alias_if_supported(client, model_name, alias, version):
    """Registry aliases exist from MLflow 2.9 and are unsupported on the
    plain file store. Treat failure as cosmetic — tags already carry the
    decision."""
    try:
        client.set_registered_model_alias(model_name, alias, version)
    except Exception as exc:
        logger.info("Registry alias not set (%s); tags remain authoritative.", exc)


def register_model_version(
    run_id: str,
    artifact_path: str,
    model_name: str = REGISTERED_MODEL_NAME,
    client: MlflowClient = None,
):
    """Register `runs:/<run_id>/<artifact_path>` as a new model version."""
    client = client or MlflowClient(tracking_uri=get_tracking_uri())
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # already exists — the normal case after the first run

    return client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id,
    )


def download_champion_artifacts(
    dst_path: str,
    model_name: str = REGISTERED_MODEL_NAME,
    client: MlflowClient = None,
):
    """Download the champion version's artifacts into `dst_path`.

    Used by the API at startup so that what is served is explicitly the
    model the registry blessed, not whatever happens to sit in models/.
    Returns (local_path, version) or (None, None) when no champion is
    available, which lets the caller fall back to the local files.
    """
    client = client or MlflowClient(tracking_uri=get_tracking_uri())
    champion = get_champion_version(model_name, client=client)
    if champion is None:
        return None, None

    os.makedirs(dst_path, exist_ok=True)
    try:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=champion.source, dst_path=dst_path
        )
    except Exception as exc:
        logger.warning("Could not download champion artifacts: %s", exc)
        return None, None

    return local, champion.version
