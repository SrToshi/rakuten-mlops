#!/usr/bin/env bash
# Airflow for this project: one container, SQLite metadata, SequentialExecutor.
#
# The production shape is Postgres + a scheduler + workers, which is three
# more containers and a lot of memory. This stack already runs six services
# on a laptop, and both DAGs are short chains of HTTP calls with nothing to
# parallelise — so the single-container form is the honest choice here, and
# the one documented as a limitation.
#
# Consequence worth knowing: with SequentialExecutor only one task runs at a
# time, which is why the training DAG's sensor uses mode="reschedule" — a
# poking sensor would hold the only slot for the hours a run takes and
# deadlock the scheduler.
set -eu

export AIRFLOW__CORE__EXECUTOR="${AIRFLOW__CORE__EXECUTOR:-SequentialExecutor}"
export AIRFLOW__CORE__LOAD_EXAMPLES="${AIRFLOW__CORE__LOAD_EXAMPLES:-False}"

echo "[airflow] applying database migrations"
airflow db migrate

echo "[airflow] ensuring the admin user exists"
airflow users create \
    --username "${AIRFLOW_ADMIN_USER:-admin}" \
    --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
    --firstname Rakuten \
    --lastname Admin \
    --role Admin \
    --email admin@example.com 2>/dev/null || echo "[airflow] admin user already present"

echo "[airflow] starting the scheduler"
airflow scheduler &

echo "[airflow] starting the webserver on 8080"
exec airflow webserver --port 8080
