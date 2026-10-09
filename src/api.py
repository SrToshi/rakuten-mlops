"""
api.py — HTTP contract for the Rakuten project.

Endpoints
    POST /predict/          score a batch of products
    POST /training/         kick off a retraining run (returns immediately)
    GET  /training/status   progress of the current or last run
    GET  /model-info        which model version is being served, and from where
    GET  /health            liveness + readiness
    GET  /metrics           Prometheus exposition

Three things here are deliberate and worth knowing before editing:

1. Models are loaded ONCE at startup, not per request. A VGG16 load takes
   seconds; repeating it on every call would dominate the response time for
   no benefit, since the weights do not change between requests.

2. What gets loaded is the version the MLflow Model Registry tagged as
   champion, downloaded at startup. If the registry is unreachable or empty,
   the local models/ directory is used instead. Serving should degrade, not
   stop, when the tracking server is down.

3. Training runs in a background thread. A full run takes hours; holding an
   HTTP connection open for that is not a design, it is an accident. The
   endpoint returns 202 immediately and /training/status reports progress.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

import db
from predict import Predict, load_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rakuten Classification API",
    description="Multimodal product classification (text + image) for Rakuten "
    "product type codes, with MLflow-backed model versioning.",
    version="2.0.0",
)

MODELS_DIR = "models"
CHAMPION_DIR = os.getenv("RAKUTEN_CHAMPION_DIR", "models/champion")
MAX_PREDICT_ROWS = int(os.getenv("RAKUTEN_MAX_PREDICT_ROWS", "200"))


# ---------------------------------------------------------------------------
# Prometheus metrics
#
# Two dashboards consume these: one for API health (the http_* series) and
# one for model behaviour and drift (the rest).
# ---------------------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "rakuten_http_requests_total",
    "HTTP requests handled, by route and outcome.",
    ["method", "endpoint", "status"],
)
HTTP_LATENCY = Histogram(
    "rakuten_http_request_duration_seconds",
    "Wall-clock time spent handling a request.",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
PREDICTIONS = Counter(
    "rakuten_predictions_total",
    "Predictions served, by predicted product type code.",
    ["prdtypecode"],
)
PREDICTION_CONFIDENCE = Histogram(
    "rakuten_prediction_confidence",
    "Confidence of the winning class.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_VERSION = Gauge(
    "rakuten_model_version",
    "Registry version of the model currently served (0 = local fallback).",
)
MODEL_LOADED = Gauge(
    "rakuten_model_loaded",
    "1 when the model is loaded and the API can serve predictions.",
)
TRAINING_IN_PROGRESS = Gauge(
    "rakuten_training_in_progress",
    "1 while a training run is executing.",
)
TRAINING_RUNS = Counter(
    "rakuten_training_runs_total",
    "Completed training runs, by outcome.",
    ["outcome"],
)
DRIFT_DETECTED = Gauge(
    "rakuten_drift_detected",
    "1 when the latest Evidently run reported dataset drift.",
)
DRIFT_SHARE = Gauge(
    "rakuten_drift_share_of_drifted_columns",
    "Share of monitored columns that drifted in the latest Evidently run.",
)
DRIFT_AGE = Gauge(
    "rakuten_drift_report_age_seconds",
    "Seconds since the latest drift report. Alerts on a stale monitor.",
)
PREDICTIONS_STORED = Gauge(
    "rakuten_predictions_stored_total",
    "Rows currently in the predictions table.",
)


# ---------------------------------------------------------------------------
# Model loading — once, at startup
# ---------------------------------------------------------------------------

STATE = {
    "artifacts": None,
    "model_version": None,
    "model_source": None,
    "loaded_at": None,
}

_load_lock = threading.Lock()


def _resolve_model_dir():
    """Prefer the registry's champion; fall back to the local directory."""
    try:
        import mlflow_utils

        local, version = mlflow_utils.download_champion_artifacts(CHAMPION_DIR)
        if local:
            logger.info("Serving champion version %s from the registry", version)
            return local, version, "mlflow-registry"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Registry lookup failed (%s); falling back to %s", exc, MODELS_DIR)

    return MODELS_DIR, None, "local-directory"


def load_model(force=False):
    """Load the serving artifacts into STATE. Safe to call concurrently."""
    with _load_lock:
        if STATE["artifacts"] is not None and not force:
            return STATE

        model_dir, version, source = _resolve_model_dir()
        artifacts = load_artifacts(model_dir)

        STATE.update(
            {
                "artifacts": artifacts,
                "model_version": version,
                "model_source": source,
                "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        MODEL_VERSION.set(float(version) if version else 0.0)
        MODEL_LOADED.set(1)
        logger.info("Model loaded from %s (%s)", model_dir, source)
        return STATE


@app.on_event("startup")
def on_startup():
    db.init_db()
    try:
        load_model()
    except Exception as exc:  # noqa: BLE001
        # A missing model must not stop the service from starting: /health
        # reports it, and /training/ can produce one.
        MODEL_LOADED.set(0)
        logger.error("Could not load a model at startup: %s", exc)


# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        HTTP_REQUESTS.labels(request.method, _route_of(request), "500").inc()
        HTTP_LATENCY.labels(_route_of(request)).observe(time.perf_counter() - started)
        raise

    endpoint = _route_of(request)
    HTTP_REQUESTS.labels(request.method, endpoint, str(status)).inc()
    HTTP_LATENCY.labels(endpoint).observe(time.perf_counter() - started)
    return response


def _route_of(request: Request) -> str:
    """Templated path, so /predict/ stays one label rather than thousands."""
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    dataset_path: str = "data/preprocessed/X_test_update.csv"
    images_path: str = "data/preprocessed/image_test"
    limit: int = 10


@app.post("/predict/")
def predict_endpoint(request: PredictRequest):
    state = load_model()
    if state["artifacts"] is None:
        raise HTTPException(status_code=503, detail="No model is loaded")

    if request.limit < 1 or request.limit > MAX_PREDICT_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be between 1 and {MAX_PREDICT_ROWS}",
        )

    predictor = Predict(
        filepath=request.dataset_path,
        imagepath=request.images_path,
        limit=request.limit,
        **state["artifacts"],
    )

    try:
        records = predictor.run()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    for record in records:
        PREDICTIONS.labels(str(record["prdtypecode"])).inc()
        PREDICTION_CONFIDENCE.observe(record["confidence"])

    _store_predictions(records, state["model_version"])

    return {
        "model_version": state["model_version"],
        "model_source": state["model_source"],
        "count": len(records),
        "predictions": records,
    }


def _store_predictions(records, model_version):
    """Persist predictions for drift monitoring.

    Wrapped because a monitoring write must never turn a successful
    prediction into a failed request.
    """
    try:
        rows = []
        for record in records:
            row = dict(record)
            row.update(db.text_features(record["designation"], record["description"]))
            rows.append(row)
        db.log_predictions(rows, model_version=model_version, source="api")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not store predictions: %s", exc)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingRequest(BaseModel):
    """Mirrors run_training()'s parameters so a demo run can stay small."""

    epochs_lstm: int = 1
    epochs_vgg: int = 1
    samples_per_class: int = 600
    batch_size: int = 32
    val_samples_per_class: int = 50
    blend_samples_per_class: int = 50
    eval_samples_per_class: int = 20


TRAINING = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "params": None,
    "result": None,
    "error": None,
}

_training_lock = threading.Lock()


def _run_training_job(params: dict):
    from training import run_training

    TRAINING_IN_PROGRESS.set(1)
    try:
        result = run_training(**params)
        TRAINING.update(
            {"status": "succeeded", "result": result, "error": None}
        )
        TRAINING_RUNS.labels("succeeded").inc()

        # Pick up whatever the registry now considers champion, so the next
        # prediction uses the model this run just produced.
        try:
            load_model(force=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Training finished but reloading the model failed: %s", exc)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        TRAINING.update({"status": "failed", "error": str(exc), "result": None})
        TRAINING_RUNS.labels("failed").inc()
    finally:
        TRAINING["finished_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        TRAINING_IN_PROGRESS.set(0)


@app.post("/training/", status_code=202)
def training_endpoint(request: TrainingRequest, background: BackgroundTasks):
    """Start a training run and return immediately.

    A full run takes hours, so the work happens on a background thread and
    the caller polls /training/status. Only one run at a time: two concurrent
    runs would race on the same files under models/.
    """
    with _training_lock:
        if TRAINING["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="A training run is already in progress",
            )
        params = request.dict()
        TRAINING.update(
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "finished_at": None,
                "params": params,
                "result": None,
                "error": None,
            }
        )

    background.add_task(_run_training_job, params)

    return {
        "status": "running",
        "params": params,
        "poll": "/training/status",
    }


@app.get("/training/status")
def training_status():
    return TRAINING


@app.post("/model/reload")
def reload_model():
    """Re-resolve the champion from the registry and load it.

    The API picks up a new champion automatically in two cases: at startup,
    and after a training run IT executed. Neither covers a run launched
    outside the service — a developer running training.py directly, a
    scheduled job, or a teammate promoting a version by hand in the MLflow
    UI. Without this endpoint the only way to pick those up is restarting the
    container, which reloads VGG16 from scratch and takes the service down
    while it does.
    """
    previous = STATE["model_version"]
    try:
        state = load_model(force=True)
    except Exception as exc:  # noqa: BLE001
        MODEL_LOADED.set(0)
        raise HTTPException(status_code=503, detail=f"Could not load a model: {exc}")

    return {
        "previous_version": previous,
        "model_version": state["model_version"],
        "model_source": state["model_source"],
        "changed": previous != state["model_version"],
        "loaded_at": state["loaded_at"],
    }


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    ready = STATE["artifacts"] is not None
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        "model_version": STATE["model_version"],
        "model_source": STATE["model_source"],
        "training": TRAINING["status"],
    }


@app.get("/model-info")
def model_info():
    return {
        "model_version": STATE["model_version"],
        "model_source": STATE["model_source"],
        "loaded_at": STATE["loaded_at"],
        "predictions_stored": _safe_count(),
        "latest_drift_report": db.latest_drift_report(),
    }


@app.get("/metrics")
def metrics():
    _refresh_monitoring_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _safe_count():
    try:
        return db.count_predictions()
    except Exception:  # noqa: BLE001
        return None


def _refresh_monitoring_gauges():
    """Pull the latest drift summary out of the database at scrape time.

    The drift job writes to the database rather than exposing its own
    endpoint: it is a batch process that exits, and Prometheus can only
    scrape something that stays up.
    """
    try:
        count = db.count_predictions()
        PREDICTIONS_STORED.set(count)
    except Exception:  # noqa: BLE001
        pass

    try:
        report = db.latest_drift_report()
        if not report:
            return
        DRIFT_DETECTED.set(float(report["dataset_drift"] or 0))
        DRIFT_SHARE.set(float(report["share_drifted_columns"] or 0))
        created = datetime.fromisoformat(report["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        DRIFT_AGE.set((datetime.now(timezone.utc) - created).total_seconds())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not refresh drift gauges: %s", exc)
