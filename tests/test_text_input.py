"""The text the model sees must be built identically at train and predict time.

Four train/serve mismatches were found in this repository; two of them lived
in how the input text and images were prepared. These tests pin the fixes so
the paths cannot silently diverge again.
"""

import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.build_features import build_text_input  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _source(relative_path):
    with open(os.path.join(SRC, relative_path), encoding="utf-8") as fh:
        return fh.read()


def test_title_and_description_are_combined():
    df = pd.DataFrame(
        {"designation": ["Cable chargeur"], "description": ["pour console DS"]}
    )
    assert build_text_input(df).tolist() == ["Cable chargeur pour console DS"]


def test_missing_description_still_yields_the_title():
    """The bug this guards: title-only products reached the model as "".

    Four different products with no description all came back as the same
    class with identical confidence, because their input text was empty.
    """
    df = pd.DataFrame(
        {
            "designation": ["Mini Turtle", "Pompe de filtration"],
            "description": [None, None],
        }
    )

    texts = build_text_input(df).tolist()
    assert texts == ["Mini Turtle", "Pompe de filtration"]
    assert texts[0] != texts[1], "distinct products must not share one input"


def test_missing_title_is_tolerated():
    df = pd.DataFrame({"designation": [None], "description": ["une description"]})
    assert build_text_input(df).tolist() == ["une description"]


def test_whole_series_is_not_stringified_into_every_row():
    """Regression guard for `designation + str(description)`.

    str() on a Series renders the entire column, so every row used to carry
    the same multi-line blob of other products' text.
    """
    df = pd.DataFrame(
        {"designation": ["A", "B", "C"], "description": ["x", "y", "z"]}
    )
    texts = build_text_input(df).tolist()

    assert texts == ["A x", "B y", "C z"]
    assert not any("\n" in text or "dtype" in text for text in texts)


def test_training_and_inference_use_the_shared_builder():
    """Neither path may assemble the text on its own."""
    for module in ("features/build_features.py", "predict.py"):
        assert "build_text_input" in _source(module), (
            f"{module} must build its text input through build_text_input()"
        )

    assert not re.search(
        r'\["designation"\]\s*\+\s*str\(', _source("features/build_features.py")
    ), "the old designation + str(description) concatenation is back"


def test_vgg16_is_trained_with_the_preprocessing_it_is_served_with():
    """Regression guard for the image-side mismatch.

    predict.py runs images through vgg16.preprocess_input. Training used a
    bare ImageDataGenerator (raw 0-255 pixels), so the image branch was
    trained on a different distribution than it was asked to score, and the
    blend search gave it a weight of 0.0.
    """
    train_src = _source("models/train_model.py")
    generators = re.findall(r"ImageDataGenerator\(([^)]*)\)", train_src)

    assert generators, "no ImageDataGenerator found in train_model.py"
    for args in generators:
        assert "preprocessing_function=preprocess_input" in args, (
            "every ImageDataGenerator must apply vgg16.preprocess_input, the "
            "same transform predict.py applies at inference time"
        )
