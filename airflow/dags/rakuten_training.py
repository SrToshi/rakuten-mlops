"""
rakuten_training — retrain the model and let the registry decide what happens.

The DAG orchestrates; it does not compute. Every task is an HTTP call to the
service that owns the work:

    start   POST /training/        the API runs the training on a background
                                   thread and answers 202 immediately
    wait    GET  /training/status  a sensor, until the run finishes
    decide  GET  /training/status  read the registry's verdict and log it
    reload  POST /model/reload     have the API pick up the new champion

Keeping the work behind HTTP is what keeps this image small: TensorFlow,
MLflow's server and Evidently each pin dependency versions that cannot share
one Python environment, which is why the project is split into services at
all. An Airflow image that imported the training code would have to re-create
that conflict.

Triggered on a schedule, and by rakuten_drift_check when drift crosses the
retraining threshold.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

API_URL = os.getenv("RAKUTEN_API_URL", "http://api:8000").rstrip("/")

# Small by default so a scheduled run finishes in minutes. A full run uses
# samples_per_class=600 and takes hours on CPU; override in the DAG's config
# when you want one ("Trigger DAG w/ config" in the UI).
DEFAULT_TRAINING_PARAMS = {
    "samples_per_class": 100,
    "epochs_lstm": 3,
    "epochs_vgg": 1,
    "val_samples_per_class": 30,
    "blend_samples_per_class": 30,
    "eval_samples_per_class": 30,
}

default_args = {
    "owner": "rakuten",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def start_training(**context):
    """Ask the API to start a run. Returns 202 without waiting for it."""
    params = dict(DEFAULT_TRAINING_PARAMS)
    params.update(context["dag_run"].conf.get("training_params", {}))

    response = requests.post(f"{API_URL}/training/", json=params, timeout=30)

    # 409 means a run is already in flight. Failing is the right outcome:
    # two concurrent runs would race on the same files under models/, and the
    # API refuses the second for that reason.
    if response.status_code == 409:
        raise AirflowFailException("A training run is already in progress")
    response.raise_for_status()

    print(f"Training started with {params}")
    return params


def training_finished(**context):
    """Sensor predicate: has the run stopped running?

    In reschedule mode the worker slot is freed between pokes, which matters
    because a full run can take hours.
    """
    status = requests.get(f"{API_URL}/training/status", timeout=30).json()
    state = status.get("status")
    print(f"Training status: {state}")
    return state in ("succeeded", "failed")


def report_outcome(**context):
    """Read the registry's decision and surface it in the task log."""
    status = requests.get(f"{API_URL}/training/status", timeout=30).json()

    if status.get("status") == "failed":
        raise AirflowFailException(f"Training failed: {status.get('error')}")

    result = status.get("result") or {}
    metrics = result.get("metrics", {})
    registry = result.get("mlflow", {})

    print(f"Weighted F1 (test): {metrics.get('ensemble_test_weighted_f1')}")
    print(f"Blend weights: {result.get('best_weights')}")

    if registry.get("promoted"):
        print(
            f"Version {registry.get('new_version')} promoted to champion "
            f"(previous: {registry.get('previous_champion_version')})"
        )
    elif registry:
        print(
            f"Version {registry.get('new_version')} kept as challenger: "
            f"{registry.get('new_metric')} did not beat "
            f"{registry.get('previous_champion_metric')}"
        )
    else:
        # Not a failure: training without tracking still produced a model.
        print("No registry decision recorded — was MLflow reachable?")

    return registry


def reload_champion(**context):
    """Have the API serve whatever the registry now considers champion."""
    response = requests.post(f"{API_URL}/model/reload", json={}, timeout=300)
    response.raise_for_status()
    result = response.json()

    if result.get("changed"):
        print(
            f"Now serving version {result['model_version']} "
            f"(was {result.get('previous_version')})"
        )
    else:
        print(f"Still serving version {result['model_version']} — no new champion")
    return result


with DAG(
    dag_id="rakuten_training",
    description="Retrain the Rakuten model; the registry decides if it ships",
    default_args=default_args,
    start_date=datetime(2026, 10, 1),
    schedule="0 3 * * 0",  # Sundays at 03:00
    catchup=False,
    max_active_runs=1,  # the API serialises runs anyway; do not queue them here
    tags=["rakuten", "training"],
) as dag:

    start = PythonOperator(
        task_id="start_training",
        python_callable=start_training,
    )

    wait = PythonSensor(
        task_id="wait_for_training",
        python_callable=training_finished,
        poke_interval=60,
        timeout=60 * 60 * 6,
        mode="reschedule",  # frees the slot between pokes
    )

    decide = PythonOperator(
        task_id="report_registry_decision",
        python_callable=report_outcome,
    )

    reload_model = PythonOperator(
        task_id="reload_champion",
        python_callable=reload_champion,
    )

    start >> wait >> decide >> reload_model
