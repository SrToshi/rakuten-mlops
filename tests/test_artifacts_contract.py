"""Guard the artifact contract between training and serving.

The project already shipped one silent failure of this kind: training wrote
models/best_weights.pkl while prediction read models/best_weights.json, so
retraining never changed a single prediction. These tests make that class of
mismatch fail loudly in CI instead of silently in production.
"""

import ast
import os
import re

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _source(relative_path):
    with open(os.path.join(SRC, relative_path), encoding="utf-8") as fh:
        return fh.read()


def _model_files_read_by(relative_path):
    """Every "models/<file>.<ext>" literal a module opens.

    The extension is required so that prose in docstrings ("files under
    models/") and directory paths ("models/champion") are not mistaken for
    artifacts.
    """
    return set(
        re.findall(r"models/([A-Za-z0-9_]+\.[A-Za-z0-9]+)", _source(relative_path))
    )


def _serving_artifacts():
    """Read SERVING_ARTIFACTS out of training.py without importing it.

    Importing would pull in TensorFlow, turning a millisecond contract check
    into a slow, environment-dependent test.
    """
    tree = ast.parse(_source("training.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SERVING_ARTIFACTS"
            for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("SERVING_ARTIFACTS not found in training.py")


def test_serving_bundle_covers_everything_inference_loads():
    """Whatever load_artifacts() opens must travel with the registered model."""
    required = _model_files_read_by("predict.py")
    missing = required - _serving_artifacts()

    assert not missing, (
        "predict.load_artifacts() opens model files that training does not "
        f"bundle into the MLflow artifact: {sorted(missing)}"
    )


def test_api_loads_artifacts_through_the_shared_loader():
    """The API must not open model files on its own.

    If it did, the CLI and the service could end up reading different files,
    which is the exact failure this suite exists to prevent.
    """
    api_src = _source("api.py")
    assert "load_artifacts" in api_src, (
        "api.py should load the model through predict.load_artifacts()"
    )

    stray = _model_files_read_by("api.py")
    assert not stray, (
        f"api.py opens model files directly instead of via load_artifacts: "
        f"{sorted(stray)}"
    )


def test_training_writes_the_weights_format_that_serving_reads():
    """The original bug: .pkl written, .json read."""
    training_src = _source("training.py")
    assert "best_weights.json" in training_src, (
        "training.py must write best_weights.json — it is the file predict.py "
        "and api.py actually read."
    )


def test_mapper_json_is_regenerated_during_training():
    """Same class of bug for the class-index mapping."""
    assert "mapper.json" in _source("features/build_features.py"), (
        "build_features.py must write mapper.json alongside mapper.pkl, or a "
        "retrained model decodes its classes with a stale mapping."
    )
