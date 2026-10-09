"""Tracking must survive its own optional extras.

A five-hour run once completed with no MLflow record at all, because the
helper that opens the run also imported data_version inside the same try
block. A missing module was therefore reported as "MLflow unavailable", the
run was created and then left orphaned, and the training carried on believing
it was tracked.

These tests pin the separation: starting the run is allowed to disable
tracking, nothing after it is.
"""

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _training_source():
    with open(os.path.join(SRC, "training.py"), encoding="utf-8") as fh:
        return fh.read()


def _function(name):
    for node in ast.walk(ast.parse(_training_source())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in training.py")


def test_optional_extras_are_not_inside_the_run_start_try():
    """data_version must not be imported where its absence disables tracking."""
    start = _function("_start_mlflow_run")

    for handler_parent in ast.walk(start):
        if not isinstance(handler_parent, ast.Try):
            continue
        # A try whose except returns the "untracked" sentinel is the one that
        # turns tracking off. Nothing optional may live inside it.
        disables_tracking = any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Tuple)
            and all(
                isinstance(element, ast.Constant) and element.value is None
                for element in node.value.elts
            )
            for handler in handler_parent.handlers
            for node in ast.walk(handler)
        )
        if not disables_tracking:
            continue

        imported = [
            alias.name
            for node in ast.walk(handler_parent)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert "data_version" not in imported, (
            "data_version is imported inside the try that disables tracking; "
            "a missing module would silently cost the run its MLflow record"
        )


def test_data_version_module_exists():
    """training.py imports it, so it has to ship alongside."""
    assert os.path.exists(os.path.join(SRC, "data_version.py")), (
        "src/data_version.py is missing but training.py imports it"
    )


def test_data_version_failure_is_reported_as_itself():
    tagger = _function("_tag_data_version")

    handled = [
        handler.type.id
        for handler in ast.walk(tagger)
        if isinstance(handler, ast.ExceptHandler) and isinstance(handler.type, ast.Name)
    ]
    assert "ImportError" in handled, (
        "_tag_data_version should name a missing module explicitly rather "
        "than folding it into a generic warning"
    )


def test_every_module_training_imports_is_present():
    """Catch the packaging mistake, not just this one instance."""
    tree = ast.parse(_training_source())
    local_modules = {
        name[:-3]
        for name in os.listdir(SRC)
        if name.endswith(".py") and name != "__init__.py"
    }

    imported_locally = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_locally |= {
                alias.name for alias in node.names if alias.name in local_modules
            }
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in local_modules:
                imported_locally.add(root)

    missing = {
        module
        for module in imported_locally
        if not os.path.exists(os.path.join(SRC, f"{module}.py"))
    }
    assert not missing, f"training.py imports modules that are not present: {missing}"


def _main_block_calls():
    """Names of the functions called directly in __main__, in order."""
    tree = ast.parse(_training_source())
    main = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and getattr(node.test.left, "id", None) == "__name__"
        ),
        None,
    )
    assert main is not None, "training.py has no __main__ block"

    names = []
    for node in main.body:
        call = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        if call is not None and isinstance(call.func, ast.Name):
            names.append(call.func.id)
    return names


def test_entry_point_reports_where_the_run_goes():
    """The CLI must name its tracking destination before training starts.

    get_tracking_uri() falls back to a local file store when
    MLFLOW_TRACKING_URI is unset. That default is deliberate, but from a
    terminal the choice is invisible: the run succeeds, registers a version,
    and the services never see it. The entry point has to say which
    destination it picked, and it has to say so before the hours of work.
    """
    calls = _main_block_calls()

    assert "_report_tracking_destination" in calls, (
        "__main__ never reports the tracking destination; a run can go to the "
        "file store with nothing on screen saying so"
    )
    assert calls.index("_report_tracking_destination") < calls.index(
        "run_training"
    ), "the destination must be reported before training starts, not after"


def test_the_report_names_the_variable_and_both_remedies():
    """A warning that does not say what to do is just noise."""
    body = ast.get_source_segment(
        _training_source(), _function("_report_tracking_destination")
    )

    assert "MLFLOW_TRACKING_URI" in body, "the report must name the variable"
    assert "/training/" in body, (
        "the report must point at the API endpoint, which is the path that "
        "needs no variable at all"
    )
