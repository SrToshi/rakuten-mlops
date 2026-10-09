"""
db.py — SQLite access layer for the Rakuten project.

Three tables live here:

  products         the source dataset, loaded once by data/import_to_db.py
  predictions      every prediction the API serves, with its input features
  drift_reports    the summary of each Evidently run

`predictions` is what turns the project from "a model behind an API" into
something monitorable: the roadmap asks to "store each prediction along with
its features in the database" precisely so a drift job can later compare the
recent traffic against the historical reference set.

Feature columns are stored alongside the raw text on purpose. Evidently needs
numeric/categorical columns to test for drift, and raw product descriptions
are neither — so the cheap derived features (length, word count) are computed
once at prediction time rather than recomputed by every consumer.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.getenv("RAKUTEN_DB_PATH", "data/rakuten.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_at    TEXT    NOT NULL,
    productid       INTEGER,
    imageid         INTEGER,
    designation     TEXT,
    description     TEXT,
    image_path      TEXT,
    text_length     INTEGER,
    word_count      INTEGER,
    prdtypecode     TEXT,
    confidence      REAL,
    model_version   TEXT,
    source          TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(predicted_at);

CREATE TABLE IF NOT EXISTS drift_reports (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT    NOT NULL,
    dataset_drift         INTEGER,
    n_drifted_columns     INTEGER,
    share_drifted_columns REAL,
    n_reference_rows      INTEGER,
    n_current_rows        INTEGER,
    mlflow_run_id         TEXT,
    report_path           TEXT
);
"""


@contextmanager
def get_connection(db_path: str = None):
    """Yield a SQLite connection with row access by column name."""
    path = db_path or DB_PATH
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = None):
    """Create the monitoring tables if they do not exist yet.

    Safe to call on every API startup: `products` is left untouched, so the
    one-time import script stays the only thing that writes the dataset.
    """
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_predictions(rows, model_version=None, source="api", db_path=None):
    """Persist a batch of predictions.

    `rows` is a list of dicts with the keys declared below. Anything missing
    is stored as NULL rather than raising: losing a monitoring row must never
    break the prediction the user actually asked for.
    """
    now = _now()
    payload = [
        (
            now,
            row.get("productid"),
            row.get("imageid"),
            row.get("designation"),
            row.get("description"),
            row.get("image_path"),
            row.get("text_length"),
            row.get("word_count"),
            str(row.get("prdtypecode")) if row.get("prdtypecode") is not None else None,
            row.get("confidence"),
            str(model_version) if model_version is not None else None,
            source,
        )
        for row in rows
    ]

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO predictions (
                predicted_at, productid, imageid, designation, description,
                image_path, text_length, word_count, prdtypecode, confidence,
                model_version, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )

    return len(payload)


def fetch_recent_predictions(limit=1000, db_path=None):
    """Most recent predictions — the 'current' dataset for drift detection."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def count_predictions(db_path=None):
    with get_connection(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def log_drift_report(
    dataset_drift,
    n_drifted_columns,
    share_drifted_columns,
    n_reference_rows,
    n_current_rows,
    mlflow_run_id=None,
    report_path=None,
    db_path=None,
):
    """Record one Evidently run so the API can expose it to Prometheus."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO drift_reports (
                created_at, dataset_drift, n_drifted_columns,
                share_drifted_columns, n_reference_rows, n_current_rows,
                mlflow_run_id, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                int(bool(dataset_drift)),
                n_drifted_columns,
                share_drifted_columns,
                n_reference_rows,
                n_current_rows,
                mlflow_run_id,
                report_path,
            ),
        )


def latest_drift_report(db_path=None):
    """Latest Evidently summary, or None if no drift job has run yet."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM drift_reports ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def fetch_reference_sample(limit=1000, db_path=None):
    """A sample of the historical dataset — the 'reference' for drift.

    Taken from the `products` table (the data the model was trained on),
    which is what makes the comparison meaningful.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT productid, imageid, designation, description, prdtypecode
            FROM products
            WHERE split = 'train'
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def text_features(designation, description):
    """Derived numeric features stored with each prediction, for drift tests."""
    text = " ".join(part for part in (designation, description) if part)
    return {"text_length": len(text), "word_count": len(text.split())}
