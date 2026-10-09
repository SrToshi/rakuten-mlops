"""Drift job: dataset assembly, and the detection itself when Evidently is present."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db  # noqa: E402
import drift_detection  # noqa: E402


@pytest.fixture()
def database(tmp_path):
    path = str(tmp_path / "rakuten.db")
    db.init_db(path)
    with db.get_connection(path) as conn:
        conn.execute(
            """CREATE TABLE products (
                   productid INTEGER, imageid INTEGER, designation TEXT,
                   description TEXT, prdtypecode TEXT, split TEXT,
                   image_path TEXT)"""
        )
        conn.executemany(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?)",
            [
                (i, i, "produit test", "une description", "2280", "train", "")
                for i in range(50)
            ],
        )
    return path


def _serve(database, n, code="2280"):
    rows = []
    for i in range(n):
        row = {
            "productid": i,
            "imageid": i,
            "designation": "produit test",
            "description": "une description",
            "prdtypecode": code,
            "confidence": 0.5,
        }
        row.update(db.text_features(row["designation"], row["description"]))
        rows.append(row)
    db.log_predictions(rows, db_path=database)


def test_reference_and_current_share_the_same_columns(database):
    """Evidently can only compare frames with matching schemas."""
    _serve(database, 10)

    reference = drift_detection.build_reference(50, db_path=database)
    current = drift_detection.build_current(10, db_path=database)

    assert list(reference.columns) == list(current.columns)
    assert len(reference) == 50
    assert len(current) == 10


def test_current_recomputes_features_for_rows_that_lack_them(database):
    """Rows written before feature storage existed must not become NaN holes."""
    db.log_predictions(
        [{"designation": "produit", "description": "texte", "prdtypecode": "10"}],
        db_path=database,
    )

    current = drift_detection.build_current(10, db_path=database)

    assert current["text_length"].notna().all()
    assert current["word_count"].notna().all()


def test_check_is_skipped_rather_than_crashing_without_traffic(database):
    """A monitor with nothing to look at reports that, it does not raise."""
    pytest.importorskip("evidently")

    outcome = drift_detection.run_drift_check(db_path=database, log_to_mlflow=False)

    assert outcome["status"] == "skipped"
    assert outcome["current_rows"] == 0


def test_summary_extraction_reads_by_key_not_position():
    """The order of metrics inside an Evidently preset is not a contract."""
    report = {
        "metrics": [
            {"metric": "SomethingElse", "result": {"unrelated": 1}},
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "dataset_drift": True,
                    "number_of_drifted_columns": 2,
                    "share_of_drifted_columns": 0.66,
                    "number_of_columns": 3,
                },
            },
        ]
    }

    summary = drift_detection._extract_summary(report)

    assert summary["dataset_drift"] is True
    assert summary["n_drifted_columns"] == 2
    assert summary["share_drifted_columns"] == pytest.approx(0.66)


def test_drift_run_persists_a_summary_and_a_report(database, tmp_path):
    pytest.importorskip("evidently")
    _serve(database, 60)

    outcome = drift_detection.run_drift_check(
        db_path=database,
        log_to_mlflow=False,
        reports_dir=str(tmp_path / "reports"),
    )

    assert outcome["status"] == "ok"
    assert os.path.exists(outcome["report_path"])

    stored = db.latest_drift_report(database)
    assert stored is not None
    assert stored["n_current_rows"] == 60
    assert stored["report_path"] == outcome["report_path"]
