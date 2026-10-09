"""
drift_service.py — HTTP wrapper around the Evidently drift check.

Why a service and not a loop
----------------------------
The drift job used to run on a sleep loop inside its own container. The brief
asks for it as an Airflow DAG instead, and a DAG needs something it can call.

Airflow could have imported drift_detection directly, but that would mean
installing Evidently — and, for the training task, TensorFlow — into the
Airflow image: exactly the dependency pile-up that forced these services
apart in the first place. So Airflow orchestrates and the services execute.
The drift check runs where Evidently already lives, behind one endpoint.

    POST /run       run a drift check, return the summary. The body is
                    optional: with none, every field falls back to its
                    default, so `curl -X POST .../run` works as typed.
    GET  /health    liveness
    GET  /latest    the most recent summary, without recomputing

The run is synchronous: a check over a few hundred rows takes seconds, and a
DAG task that waits for its own result is simpler to reason about than one
that polls. Training is the opposite case and is handled the other way, in
the API.
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
import drift_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rakuten drift service",
    description="Evidently data-drift checks, callable from an Airflow DAG.",
    version="1.0.0",
)


class DriftRequest(BaseModel):
    reference_limit: int = 1000
    current_limit: int = 500
    log_to_mlflow: bool = True


@app.on_event("startup")
def on_startup():
    # The predictions and drift_reports tables may not exist yet if this
    # container starts before anything has served a prediction.
    try:
        db.init_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not initialise the database: %s", exc)


@app.get("/health")
def health():
    return {"status": "ok", "reports_dir": drift_detection.REPORTS_DIR}


@app.post("/run")
def run(request: DriftRequest = DriftRequest()):
    """Run one drift check and return its summary.

    `status: "skipped"` (not an error) means there was nothing to compare
    yet — no predictions have been served. The DAG treats that as a no-op
    rather than a failure, because an idle system is not a broken one.

    The body is optional. Every field of DriftRequest already has a default,
    so requiring the body contradicted the model: `curl -X POST .../run` —
    the obvious thing to type, and what the README told people to type —
    answered 422 complaining about a missing field that has a default. The
    DAG always sends all three, so nothing automated ever hit this; only a
    person at a terminal did, which is the worst place to find out.
    """
    try:
        summary = drift_detection.run_drift_check(
            reference_limit=request.reference_limit,
            current_limit=request.current_limit,
            log_to_mlflow=request.log_to_mlflow,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Drift check failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return summary


@app.get("/latest")
def latest():
    """The last stored summary. Lets a DAG or a dashboard read the verdict
    without paying for a fresh comparison."""
    report = db.latest_drift_report()
    if report is None:
        return {"status": "none", "detail": "no drift check has run yet"}
    return report


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DRIFT_SERVICE_PORT", "8100")),
    )
