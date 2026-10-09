"""Monitoring tables: prediction logging and drift bookkeeping."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import db  # noqa: E402


@pytest.fixture()
def database(tmp_path):
    path = str(tmp_path / "rakuten.db")
    db.init_db(path)
    return path


def test_init_is_idempotent(database):
    db.init_db(database)  # must not raise on an existing schema
    assert db.count_predictions(database) == 0


def test_predictions_round_trip(database):
    rows = [
        {
            "productid": 1,
            "imageid": 10,
            "designation": "Cable chargeur",
            "description": "pour console",
            "image_path": "data/preprocessed/image_test/image_10_product_1.jpg",
            "text_length": 27,
            "word_count": 4,
            "prdtypecode": "1280",
            "confidence": 0.81,
        }
    ]
    written = db.log_predictions(rows, model_version="3", source="api", db_path=database)

    assert written == 1
    assert db.count_predictions(database) == 1

    stored = db.fetch_recent_predictions(db_path=database)[0]
    assert stored["prdtypecode"] == "1280"
    assert stored["model_version"] == "3"
    assert stored["source"] == "api"
    assert stored["predicted_at"]


def test_missing_fields_become_null_instead_of_raising(database):
    """A monitoring row must never be able to break a served prediction."""
    db.log_predictions([{"prdtypecode": "2280"}], db_path=database)

    stored = db.fetch_recent_predictions(db_path=database)[0]
    assert stored["prdtypecode"] == "2280"
    assert stored["productid"] is None
    assert stored["confidence"] is None


def test_recent_predictions_are_returned_newest_first(database):
    db.log_predictions([{"prdtypecode": "A"}], db_path=database)
    db.log_predictions([{"prdtypecode": "B"}], db_path=database)

    codes = [row["prdtypecode"] for row in db.fetch_recent_predictions(db_path=database)]
    assert codes == ["B", "A"]


def test_drift_report_round_trip(database):
    assert db.latest_drift_report(database) is None

    db.log_drift_report(
        dataset_drift=True,
        n_drifted_columns=2,
        share_drifted_columns=0.5,
        n_reference_rows=1000,
        n_current_rows=200,
        mlflow_run_id="abc123",
        db_path=database,
    )

    report = db.latest_drift_report(database)
    assert report["dataset_drift"] == 1
    assert report["share_drifted_columns"] == 0.5
    assert report["mlflow_run_id"] == "abc123"


def test_text_features():
    assert db.text_features("Cable", "chargeur console") == {
        "text_length": 22,
        "word_count": 3,
    }
    # Missing description must not crash or poison the numbers.
    assert db.text_features("Cable", None) == {"text_length": 5, "word_count": 1}
