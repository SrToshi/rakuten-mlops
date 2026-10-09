"""The DAGs must orchestrate and compute nothing.

That sentence is the whole architecture. TensorFlow pins typing_extensions
below what MLflow's server needs, and Evidently brings its own web stack;
those sets cannot share one Python environment, which is why the project is
split into services. An Airflow image that imported the training code — or
Evidently — would re-create the conflict that forced the split, and it would
do so quietly, at build time, weeks after anyone remembered why.

So these tests read the DAG files as source rather than running them:

  * no heavy import may appear in a DAG module,
  * the Airflow image may install no heavy dependency,
  * every endpoint a DAG calls must exist in the service that serves it,
  * the drift DAG must trigger the DAG that actually exists.

They need no Airflow installed, which is the point: the dev environment has
no Airflow either, for exactly the same reason. The last test does use a real
DagBag, and skips when Airflow is absent.
"""

import ast
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
DAGS_DIR = os.path.join(ROOT, "airflow", "dags")
SRC = os.path.join(ROOT, "src")

DAG_FILES = ["rakuten_training.py", "rakuten_drift_check.py"]

# Anything whose presence in the orchestrator would mean it is doing the work
# itself, plus the two modules that carry the dependency conflict.
FORBIDDEN_IMPORTS = {
    "tensorflow",
    "keras",
    "evidently",
    "sklearn",
    "scikit-learn",
    "training",
    "drift_detection",
    "predict",
    "mlflow",
}


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _imported_roots(source):
    """Every top-level module name a file imports."""
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("filename", DAG_FILES)
def test_dag_files_exist(filename):
    assert os.path.exists(os.path.join(DAGS_DIR, filename))


@pytest.mark.parametrize("filename", DAG_FILES)
def test_dags_import_nothing_heavy(filename):
    """The orchestrator calls services; it does not load their dependencies."""
    roots = _imported_roots(_read(os.path.join(DAGS_DIR, filename)))
    offenders = roots & FORBIDDEN_IMPORTS
    assert not offenders, (
        f"{filename} imports {offenders}. Airflow decides WHEN things run and "
        "the services decide HOW; importing the work here would put the "
        "conflicting dependency sets back into one environment."
    )


def test_airflow_image_installs_no_heavy_dependencies():
    """Same rule, enforced at the image rather than the module."""
    # Comments are stripped first: the file explains at length which packages
    # it refuses to install, and naming them is not installing them.
    dockerfile = "\n".join(
        line for line in _read(os.path.join(ROOT, "docker", "Dockerfile.airflow")).lower().splitlines()
        if not line.strip().startswith("#")
    )

    for package in ("tensorflow", "evidently", "scikit-learn"):
        assert package not in dockerfile, (
            f"Dockerfile.airflow mentions {package}; the Airflow image must "
            "stay free of the heavy dependency sets"
        )

    # mlflow-skinny would be acceptable (it is only a client), the server is
    # not: it is the SQLAlchemy/Alembic pile that started the whole conflict.
    assert not re.search(r"\bmlflow\b(?!-skinny)", dockerfile), (
        "Dockerfile.airflow installs the MLflow server; the DAGs read the "
        "registry's verdict through the API instead"
    )


def test_endpoints_the_dags_call_exist():
    """A DAG that posts to a route the service does not expose fails at 03:00
    on a Sunday, in a log nobody is reading. Check it here instead."""
    api_source = _read(os.path.join(SRC, "api.py"))
    drift_source = _read(os.path.join(SRC, "drift_service.py"))

    api_routes = set(re.findall(r'@app\.\w+\(\s*"([^"]+)"', api_source))
    drift_routes = set(re.findall(r'@app\.\w+\(\s*"([^"]+)"', drift_source))

    for route in ("/training/", "/training/status", "/model/reload"):
        assert route in api_routes, f"the training DAG calls {route}, api.py does not serve it"

    assert "/run" in drift_routes, "the drift DAG posts to /run, drift_service.py does not serve it"


def test_drift_dag_triggers_a_dag_that_exists():
    """TriggerDagRunOperator takes a string. A typo in it is silent."""
    source = _read(os.path.join(DAGS_DIR, "rakuten_drift_check.py"))

    triggered = re.search(r'trigger_dag_id\s*=\s*"([^"]+)"', source)
    assert triggered, "rakuten_drift_check does not name a DAG to trigger"

    training_source = _read(os.path.join(DAGS_DIR, "rakuten_training.py"))
    declared = re.search(r'dag_id\s*=\s*"([^"]+)"', training_source)
    assert declared, "rakuten_training does not declare a dag_id"

    assert triggered.group(1) == declared.group(1), (
        f"the drift DAG triggers {triggered.group(1)!r} but the training DAG "
        f"is called {declared.group(1)!r}"
    )


def test_training_sensor_reschedules_rather_than_holding_a_slot():
    """A full run takes hours. A poking sensor would occupy the only worker
    slot the SequentialExecutor has, and nothing else would ever run."""
    source = _read(os.path.join(DAGS_DIR, "rakuten_training.py"))
    assert 'mode="reschedule"' in source, (
        "wait_for_training should reschedule between pokes; in poke mode it "
        "would hold the executor's only slot for the whole run"
    )


def test_drift_trigger_does_not_wait_for_the_training_run():
    """The hourly drift check must not stall for the hours training takes."""
    source = _read(os.path.join(DAGS_DIR, "rakuten_drift_check.py"))
    assert "wait_for_completion=False" in source, (
        "the drift DAG should fire training and move on, or the hourly "
        "schedule would back up behind a multi-hour run"
    )


def test_dags_parse_under_real_airflow():
    """The checks above read source. This one builds the graph.

    Skipped wherever Airflow is not installed, which includes the dev
    environment — the Airflow image is the only place it lives.
    """
    # Not importorskip("airflow"): the repo has an airflow/ directory, which
    # Python happily treats as an empty namespace package, so the bare name
    # imports even with nothing installed. Ask for the real submodule.
    pytest.importorskip(
        "airflow.models", reason="Airflow is only installed in its own image"
    )

    from airflow.models import DagBag

    bag = DagBag(dag_folder=DAGS_DIR, include_examples=False)
    assert not bag.import_errors, bag.import_errors

    assert set(bag.dag_ids) == {"rakuten_training", "rakuten_drift_check"}

    # bag.dags, not bag.get_dag(): the latter consults the metadata
    # database, which a parse-only check has no reason to need.
    training = bag.dags["rakuten_training"]
    assert [task.task_id for task in training.topological_sort()] == [
        "start_training",
        "wait_for_training",
        "report_registry_decision",
        "reload_champion",
    ]

    drift = bag.dags["rakuten_drift_check"]
    assert [task.task_id for task in drift.topological_sort()] == [
        "run_drift_check",
        "drift_above_threshold",
        "trigger_training",
    ]
