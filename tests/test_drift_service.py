"""The drift service's HTTP contract.

Two tests, for one papercut that only ever bites a human.

`DriftRequest` gives all three of its fields a default, but FastAPI makes a
Pydantic body parameter required unless the parameter itself has a default.
So `POST /run` with no body answered 422, complaining about a missing field
that has a default — while the README, and the closing line of
simulate_traffic.py, both told people to type exactly that.

The DAG always sends all three fields, so nothing automated ever touched this
path. It was reachable only by a person typing the obvious command at a
terminal, which is the worst place and time to discover it.

The Evidently comparison is replaced by a double here on purpose. What is
being pinned is which values reach `run_drift_check` — the defaults when no
body is sent, the overrides when one is — not what Evidently concludes about
them. That has its own tests, and a real comparison would need a populated
`products` table that `db.init_db` does not create.
"""

import os
import sys

import pytest

pytest.importorskip(
    "evidently", reason="the drift service's dependencies live in its own image"
)

from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import drift_service  # noqa: E402

# What DriftRequest declares. If these change, the service's defaults change
# with them, and anyone typing the bare command gets a different window.
DEFAULTS = {"reference_limit": 1000, "current_limit": 500, "log_to_mlflow": True}


@pytest.fixture
def received(monkeypatch):
    """Capture the arguments the endpoint passes on, without running a check."""
    captured = {}

    def fake_run_drift_check(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "share_of_drifted_columns": 0.0}

    monkeypatch.setattr(
        drift_service.drift_detection, "run_drift_check", fake_run_drift_check
    )
    return captured


@pytest.fixture
def client():
    return TestClient(drift_service.app)


def test_run_accepts_a_request_with_no_body(client, received):
    """`curl -X POST .../run` must work as typed, and use the defaults."""
    response = client.post("/run")

    assert response.status_code != 422, (
        "POST /run rejected an empty body. Every DriftRequest field has a "
        "default, so the body must stay optional: the parameter needs its own "
        "default (`request: DriftRequest = DriftRequest()`), or FastAPI makes "
        "the body required and the obvious command fails for a person at a "
        "terminal — the one caller that cannot read the error and retry."
    )
    assert response.status_code == 200, response.text
    assert received == DEFAULTS


def test_run_still_accepts_the_limits_the_dag_sends(client, received):
    """The DAG sends all three fields; overriding must keep working."""
    sent = {"reference_limit": 50, "current_limit": 20, "log_to_mlflow": False}

    response = client.post("/run", json=sent)

    assert response.status_code == 200, response.text
    assert received == sent


def test_a_partial_body_fills_the_rest_from_the_defaults(client, received):
    """Narrowing just the current window is the useful demo gesture:

        curl -X POST .../run -d '{"current_limit": 120}'

    so the window covers only the traffic just sent, rather than several
    sessions of it.
    """
    response = client.post("/run", json={"current_limit": 120})

    assert response.status_code == 200, response.text
    assert received == {**DEFAULTS, "current_limit": 120}
