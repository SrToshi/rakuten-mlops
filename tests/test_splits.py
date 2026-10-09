"""The three data splits must not overlap.

Validation used to be sampled out of the test set and never removed from it,
so the metric that decided champion versus challenger was computed on rows
the networks had already seen through validation_data and EarlyStopping.
These tests make that impossible to reintroduce unnoticed.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.build_features import DataImporter  # noqa: E402

N_CLASSES = 5
ROWS_PER_CLASS = 40


@pytest.fixture()
def dataset():
    rows = []
    identifier = 0
    for class_label in range(N_CLASSES):
        for _ in range(ROWS_PER_CLASS):
            rows.append(
                {
                    "description": f"produit {identifier}",
                    "productid": identifier,
                    "imageid": identifier,
                    "prdtypecode": class_label,
                }
            )
            identifier += 1
    return pd.DataFrame(rows)


def _split(dataset, **kwargs):
    return DataImporter().split_train_test(dataset, **kwargs)


def test_splits_are_disjoint(dataset):
    X_train, X_val, X_test, _, _, _ = _split(
        dataset, samples_per_class=10, val_samples_per_class=5
    )

    train_ids = set(X_train["productid"])
    val_ids = set(X_val["productid"])
    test_ids = set(X_test["productid"])

    assert not train_ids & val_ids, "training rows leaked into validation"
    assert not train_ids & test_ids, "training rows leaked into test"
    assert not val_ids & test_ids, (
        "validation rows leaked into test — the promotion metric would be "
        "measured on rows already used for model selection"
    )


def test_every_row_is_used_exactly_once(dataset):
    X_train, X_val, X_test, _, _, _ = _split(
        dataset, samples_per_class=10, val_samples_per_class=5
    )

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(dataset)


def test_each_split_is_class_balanced(dataset):
    _, _, _, y_train, y_val, _ = _split(
        dataset, samples_per_class=10, val_samples_per_class=5
    )

    assert y_train.value_counts().nunique() == 1
    assert y_train.value_counts().iloc[0] == 10
    assert y_val.value_counts().iloc[0] == 5


def test_features_stay_aligned_with_their_labels(dataset):
    """productid encodes the class, so misalignment is detectable."""
    X_train, X_val, X_test, y_train, y_val, y_test = _split(
        dataset, samples_per_class=10, val_samples_per_class=5
    )

    for features, labels in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        expected = features["productid"] // ROWS_PER_CLASS
        assert (expected.values == labels.values).all(), (
            "a split's labels are no longer aligned with its features"
        )


def test_test_set_can_be_capped(dataset):
    """Each test row costs a VGG16 forward pass, so the size must be bounded."""
    _, _, X_test, _, _, _ = _split(
        dataset,
        samples_per_class=10,
        val_samples_per_class=5,
        test_samples_per_class=3,
    )

    assert len(X_test) == N_CLASSES * 3


def test_requesting_more_validation_than_available_does_not_raise(dataset):
    _, X_val, X_test, _, _, _ = _split(
        dataset, samples_per_class=35, val_samples_per_class=20
    )

    # 40 - 35 = 5 rows left per class, so validation is clamped to those.
    assert len(X_val) == N_CLASSES * 5
    assert len(X_test) == 0
