"""
import_to_db.py — one-time script to load the Rakuten CSVs into a local
SQLite database.

Run once, from the project root, with the 'rakuten-mlops' conda env active:
    python src/data/import_to_db.py

Reads the three preprocessed CSVs (X_train, Y_train, X_test) and writes a
single `products` table to data/rakuten.db, so downstream training and
serving code have one consistent source of truth instead of three loose
files.

Design choices:
- Only tabular data (ids, text, category) goes into the database.
  Images stay on disk as files; only their relative path is stored.
- designation/description are stored raw, exactly as provided by the
  challenge dataset — no feature engineering here. That stays the
  responsibility of features/build_features.py, applied at train/predict
  time.
- prdtypecode is the ORIGINAL Rakuten category code (e.g. 1280, 2280),
  not the internal 0-26 index used during training. It is stored as
  ground truth, independent of whichever label-encoding a given training
  run produces.
- Test rows have prdtypecode = NULL, since their true label isn't known.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/preprocessed")
DB_PATH = Path("data/rakuten.db")

# Original filenames as provided by the Rakuten/ENS challenge platform,
# kept as-is (including the opaque "CVw08PX" suffix) to match what
# features/build_features.py's DataImporter already expects. Renaming
# would require updating that shared module too, which is out of scope
# for this one-time import script.
X_TRAIN_FILE = "X_train_update.csv"
Y_TRAIN_FILE = "Y_train_CVw08PX.csv"
X_TEST_FILE = "X_test_update.csv"


def _read_csv_drop_unnamed_index(path: Path) -> pd.DataFrame:
    """Read a CSV and drop the stray 'Unnamed: 0' index column if present."""
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


def build_image_path(df: pd.DataFrame, image_dir: str) -> pd.Series:
    return (
        image_dir
        + "/image_"
        + df["imageid"].astype(str)
        + "_product_"
        + df["productid"].astype(str)
        + ".jpg"
    )


def load_train() -> pd.DataFrame:
    X = _read_csv_drop_unnamed_index(DATA_DIR / X_TRAIN_FILE)
    y = _read_csv_drop_unnamed_index(DATA_DIR / Y_TRAIN_FILE)
    # Same row order in both files -> safe to align side by side (as the
    # original DataImporter.load_data() does with pd.concat(axis=1)).
    df = pd.concat([X, y], axis=1)
    df["split"] = "train"
    df["image_path"] = build_image_path(df, "data/preprocessed/image_train")
    return df


def load_test() -> pd.DataFrame:
    X = _read_csv_drop_unnamed_index(DATA_DIR / X_TEST_FILE)
    X["prdtypecode"] = None
    X["split"] = "test"
    X["image_path"] = build_image_path(X, "data/preprocessed/image_test")
    return X


def main():
    train_df = load_train()
    test_df = load_test()
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    full_df = full_df[
        ["productid", "imageid", "designation", "description",
         "prdtypecode", "split", "image_path"]
    ]

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    full_df.to_sql("products", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded {len(full_df)} rows into {DB_PATH} (table 'products')")
    print(f"  train: {(full_df['split'] == 'train').sum()}")
    print(f"  test:  {(full_df['split'] == 'test').sum()}")


if __name__ == "__main__":
    main()