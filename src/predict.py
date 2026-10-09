"""
predict.py — batch inference for the Rakuten fusion model.

Predict.run() returns one record per row instead of a bare {index: code}
mapping. Two reasons:

  * the API has to persist each prediction with its input features, which
    means it needs the inputs and the confidence, not just the label;
  * the number of rows was hardcoded to the first 10 of the CSV, so the
    script silently ignored everything else it was pointed at.

main() still writes data/preprocessed/predictions.json in the original
{index: prdtypecode} shape, so anything built against the old output keeps
working.
"""

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

from features.build_features import (
    ImagePreprocessor,
    TextPreprocessor,
    build_text_input,
)

DEFAULT_LIMIT = 10


class Predict:
    def __init__(
        self,
        tokenizer,
        lstm,
        vgg16,
        best_weights,
        mapper,
        filepath,
        imagepath,
        limit=DEFAULT_LIMIT,
    ):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper
        self.filepath = filepath
        self.imagepath = imagepath
        # None means "the whole file". Keeping a default avoids someone
        # accidentally running inference over 84k images from a web request.
        self.limit = limit

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def run(self):
        """Return one record per row: inputs, predicted class and confidence."""
        X = pd.read_csv(self.filepath)
        if self.limit:
            X = X.head(self.limit)
        X = X.copy()

        # Keep the text as submitted. The preprocessor below overwrites
        # `description` in place with its lemmatised form, which is what the
        # model wants but not what we should store or show back to a caller.
        raw_designation = X.get("designation", pd.Series([None] * len(X))).tolist()
        raw_description = X.get("description", pd.Series([None] * len(X))).tolist()

        # Build the model's text input exactly as training does, from the
        # product title AND its description. Passing description alone sent
        # title-only products to the model as an empty string.
        X["description"] = build_text_input(X)

        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor(self.imagepath)
        text_preprocessor.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)

        sequences = self.tokenizer.texts_to_sequences(X["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )

        target_size = (224, 224, 3)
        images = X["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        lstm_proba = self.lstm.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([images])

        concatenate_proba = (
            self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)
        confidences = np.max(concatenate_proba, axis=1)

        records = []
        for position, class_index in enumerate(final_predictions):
            records.append(
                {
                    "index": position,
                    "productid": _safe_int(X["productid"].iloc[position]),
                    "imageid": _safe_int(X["imageid"].iloc[position]),
                    "designation": _safe_str(raw_designation[position]),
                    "description": _safe_str(raw_description[position]),
                    "image_path": X["image_path"].iloc[position],
                    "prdtypecode": self.mapper[str(class_index)],
                    "confidence": float(confidences[position]),
                }
            )
        return records

    def predict(self):
        """Backwards-compatible {index: prdtypecode} mapping."""
        return {
            str(record["index"]): record["prdtypecode"] for record in self.run()
        }


def _safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def load_artifacts(models_dir="models"):
    """Load every artifact an inference needs, from one directory.

    Shared with api.py so that the CLI and the service can never drift onto
    different files.
    """
    with open(f"{models_dir}/tokenizer_config.json", "r", encoding="utf-8") as fh:
        tokenizer = keras.preprocessing.text.tokenizer_from_json(fh.read())

    lstm = keras.models.load_model(f"{models_dir}/best_lstm_model.h5")
    vgg16 = keras.models.load_model(f"{models_dir}/best_vgg16_model.h5")

    with open(f"{models_dir}/best_weights.json", "r") as fh:
        best_weights = json.load(fh)

    with open(f"{models_dir}/mapper.json", "r") as fh:
        mapper = json.load(fh)

    return {
        "tokenizer": tokenizer,
        "lstm": lstm,
        "vgg16": vgg16,
        "best_weights": best_weights,
        "mapper": mapper,
    }


def main():
    parser = argparse.ArgumentParser(description="Input data")
    parser.add_argument(
        "--dataset_path",
        default="data/preprocessed/X_train_update.csv",
        type=str,
        help="File path for the input CSV file.",
    )
    parser.add_argument(
        "--images_path",
        default="data/preprocessed/image_train",
        type=str,
        help="Base path for the images.",
    )
    parser.add_argument(
        "--limit",
        default=DEFAULT_LIMIT,
        type=int,
        help="How many rows to score. 0 means the whole file.",
    )
    args = parser.parse_args()

    artifacts = load_artifacts()
    predictor = Predict(
        filepath=args.dataset_path,
        imagepath=args.images_path,
        limit=args.limit or None,
        **artifacts,
    )

    predictions = predictor.predict()

    with open("data/preprocessed/predictions.json", "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)


if __name__ == "__main__":
    main()
