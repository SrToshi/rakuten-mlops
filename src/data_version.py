"""
data_version.py — read DVC data hashes so training runs can record them.

The roadmap asks to "use DVC (without Git) to version datasets and store
their hashes in MLflow". This module is the second half of that: it reads the
`.dvc` pointer files DVC produces and hands the hashes to training.py, which
logs them as MLflow tags.

Why that matters: an MLflow run already records the code parameters and the
resulting metrics, but not *which data* produced them. Two runs with
identical parameters and different metrics are unexplainable without it. With
the hash on the run, a disagreement between runs can always be traced to
either the code or the data.

DVC is initialised without Git (`dvc init --no-scm`) as the brief specifies,
so the .dvc files are the record rather than Git commits. See `make_dvc.py`
for the setup commands.
"""

import logging
import os

logger = logging.getLogger(__name__)

# The .dvc pointer files whose hashes identify the training data.
TRACKED_PATHS = [
    "data/preprocessed/X_train_update.csv.dvc",
    "data/preprocessed/Y_train_CVw08PX.csv.dvc",
    "data/preprocessed/X_test_update.csv.dvc",
    "data/preprocessed/image_train.dvc",
]


def _parse_dvc_file(path):
    """Extract the md5 from a .dvc pointer.

    Parsed by hand rather than with PyYAML so that the training image does not
    need a YAML dependency just to read four short files. The format is
    stable: a list of outs, each with md5/size/path keys.
    """
    entry = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip().lstrip("- ").strip()
                for key in ("md5", "size", "nfiles", "path"):
                    prefix = f"{key}:"
                    if stripped.startswith(prefix):
                        entry[key] = stripped[len(prefix):].strip()
    except OSError:
        return None
    return entry or None


def collect_data_hashes(paths=None, root="."):
    """Return {dataset name: md5} for every .dvc pointer that exists.

    Missing pointers are skipped silently: DVC is optional tooling, and a
    developer who has not run `dvc add` should still be able to train.
    """
    hashes = {}
    for relative in paths or TRACKED_PATHS:
        full = os.path.join(root, relative)
        if not os.path.exists(full):
            continue
        entry = _parse_dvc_file(full)
        if not entry or "md5" not in entry:
            continue
        name = os.path.basename(relative)[: -len(".dvc")]
        hashes[name] = entry["md5"]
    return hashes


def mlflow_tags(paths=None, root="."):
    """Data hashes as MLflow tags, prefixed so they group in the UI."""
    hashes = collect_data_hashes(paths=paths, root=root)
    if not hashes:
        logger.info("No .dvc pointers found — data version not recorded")
    return {f"data.{name}": md5 for name, md5 in hashes.items()}
