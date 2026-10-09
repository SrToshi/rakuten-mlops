"""DVC pointer parsing — the link between a dataset version and an MLflow run."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_version  # noqa: E402

POINTER = """outs:
- md5: 7f2c1a9e4b8d6f3a2c5e9b1d4f7a0c3e
  size: 54129120
  hash: md5
  path: X_train_update.csv
"""

DIRECTORY_POINTER = """outs:
- md5: aa11bb22cc33dd44ee55ff66aa77bb88.dir
  size: 2411724800
  nfiles: 84916
  hash: md5
  path: image_train
"""


def _write(tmp_path, name, content):
    target = tmp_path / "data" / "preprocessed"
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text(content)


def test_reads_a_file_pointer(tmp_path):
    _write(tmp_path, "X_train_update.csv.dvc", POINTER)

    hashes = data_version.collect_data_hashes(root=str(tmp_path))

    assert hashes == {"X_train_update.csv": "7f2c1a9e4b8d6f3a2c5e9b1d4f7a0c3e"}


def test_reads_a_directory_pointer(tmp_path):
    """Directory hashes carry a .dir suffix and must survive intact."""
    _write(tmp_path, "image_train.dvc", DIRECTORY_POINTER)

    hashes = data_version.collect_data_hashes(root=str(tmp_path))

    assert hashes["image_train"] == "aa11bb22cc33dd44ee55ff66aa77bb88.dir"


def test_missing_pointers_are_skipped_not_fatal(tmp_path):
    """DVC is optional tooling: training must work without it."""
    assert data_version.collect_data_hashes(root=str(tmp_path)) == {}
    assert data_version.mlflow_tags(root=str(tmp_path)) == {}


def test_tags_are_namespaced(tmp_path):
    _write(tmp_path, "X_train_update.csv.dvc", POINTER)

    tags = data_version.mlflow_tags(root=str(tmp_path))

    assert tags == {"data.X_train_update.csv": "7f2c1a9e4b8d6f3a2c5e9b1d4f7a0c3e"}
