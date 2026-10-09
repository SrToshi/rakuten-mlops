"""Champion/challenger promotion logic.

These run without TensorFlow and without the dataset: they exercise the
registry decision rules against a throwaway SQLite-backed MLflow store,
which is what makes them usable as a CI gate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

mlflow = pytest.importorskip("mlflow")
import mlflow_utils  # noqa: E402


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """Isolated MLflow tracking store + registry for one test.

    Uses the file store, which is what the training environment actually
    runs (mlflow-skinny has no SQLAlchemy, so it cannot open a sqlite://
    URI). MLFLOW_ALLOW_FILE_STORE keeps this working on MLflow 3, where the
    file backend is deprecated but still functional.
    """
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    monkeypatch.chdir(tmp_path)
    mlflow_utils.setup_experiment("tests")
    yield
    try:
        mlflow.end_run()
    except Exception:
        pass


def _training_run(weighted_f1, tmp_path):
    """Simulate one training run: log a metric and a serving bundle."""
    with mlflow.start_run() as run:
        mlflow.log_metric(mlflow_utils.PRIMARY_METRIC, weighted_f1)
        bundle = tmp_path / f"bundle_{run.info.run_id}"
        bundle.mkdir()
        (bundle / "best_weights.json").write_text("[0.5, 0.5]")
        mlflow.log_artifacts(str(bundle), artifact_path="model")
        run_id = run.info.run_id

    version = mlflow_utils.register_model_version(run_id, "model")
    return mlflow_utils.promote_if_better(version)


def test_first_model_becomes_champion(registry, tmp_path):
    result = _training_run(0.60, tmp_path)
    assert result["promoted"] is True
    assert mlflow_utils.get_champion_version().version == result["new_version"]


def test_worse_model_is_not_promoted(registry, tmp_path):
    _training_run(0.60, tmp_path)
    result = _training_run(0.55, tmp_path)

    assert result["promoted"] is False
    # The reigning champion must be untouched by a losing challenger.
    assert mlflow_utils.get_champion_version().version != result["new_version"]


def test_better_model_replaces_champion(registry, tmp_path):
    first = _training_run(0.60, tmp_path)
    second = _training_run(0.71, tmp_path)

    assert second["promoted"] is True
    assert second["previous_champion_version"] == first["new_version"]

    champion = mlflow_utils.get_champion_version()
    assert champion.version == second["new_version"]
    assert champion.tags["stage"] == "champion"


def test_champion_artifacts_are_downloadable(registry, tmp_path):
    _training_run(0.60, tmp_path)
    local, version = mlflow_utils.download_champion_artifacts(str(tmp_path / "dl"))

    assert version is not None
    assert "best_weights.json" in os.listdir(local)
