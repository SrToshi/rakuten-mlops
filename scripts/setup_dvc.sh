#!/usr/bin/env bash
# Initialise DVC and start tracking the Rakuten datasets.
#
# Run once, from the repository root:
#     bash scripts/setup_dvc.sh            # CSVs only (fast)
#     bash scripts/setup_dvc.sh --images   # CSVs + the 84,916 training images
#
# On Windows, run the same commands by hand in cmd — they are identical
# apart from the shell syntax.
#
# `dvc init --no-scm` is deliberate: the brief asks for DVC *without* Git, so
# the .dvc pointer files are themselves the record of which data version was
# used, rather than a Git commit. training.py reads those pointers and tags
# each MLflow run with the hashes (see src/data_version.py), which is what
# makes a run reproducible: parameters from MLflow, data from DVC.
#
# Note on disk: `dvc add` copies tracked data into .dvc/cache, so adding the
# images costs roughly another 2.4 GB. That is the price of being able to
# restore an exact dataset version, and it is why the images are opt-in here.
#
# Second consequence of --no-scm: DVC does NOT write the .gitignore files it
# normally drops beside its cache and its tracked data. The entries for
# .dvc/cache/ and .dvc/tmp/ are hand-written in the repository's .gitignore;
# without them, `git add -A` would sweep the cache into a commit.

set -eu

TRACK_IMAGES=0
[ "${1:-}" = "--images" ] && TRACK_IMAGES=1

if ! command -v dvc >/dev/null 2>&1; then
    echo "DVC is not installed. Put it in an environment of its own: dvc 3.x"
    echo "upgrades typing_extensions past the < 4.6 that TensorFlow 2.13 pins,"
    echo "so it must never share the training environment."
    echo
    echo "    python -m venv .venv-dvc"
    echo "    . .venv-dvc/bin/activate      # Windows: .venv-dvc\\Scripts\\activate"
    echo "    pip install 'dvc==3.55.2' 'pathspec==0.12.1'"
    echo
    echo "The pathspec pin is not optional. dvc 3.55.2 requires pathspec with"
    echo "no upper bound, so pip resolves it to 1.x, which dropped the private"
    echo "_DIR_MARK that dvc's ignore handling imports. Every dvc command then"
    echo "fails identically, before doing any work:"
    echo
    echo "    ERROR: unexpected error - cannot import name '_DIR_MARK'"
    echo "    from 'pathspec.patterns.gitwildmatch'"
    exit 1
fi

if [ ! -d .dvc ]; then
    echo "==> dvc init --no-scm"
    dvc init --no-scm
fi

echo "==> tracking the tabular data"
dvc add data/preprocessed/X_train_update.csv
dvc add data/preprocessed/Y_train_CVw08PX.csv
dvc add data/preprocessed/X_test_update.csv

if [ "$TRACK_IMAGES" = "1" ]; then
    echo "==> tracking the training images (slow: ~85k files)"
    dvc add data/preprocessed/image_train
else
    echo "==> skipping images (pass --images to include them)"
fi

echo
echo "Done. The .dvc pointer files now identify this data version:"
ls -1 data/preprocessed/*.dvc 2>/dev/null || true
echo
echo "The next training run will tag its MLflow run with these hashes."
echo
echo "---"
echo "Sharing the data with the team (optional): add a remote."
echo
echo "Without one, the versioning is local: a teammate who clones gets the"
echo "pointers and the hashes, but nothing to retrieve the data with."
echo
echo "    pip install 'dvc[gdrive]==3.55.2'"
echo "    dvc remote add -d drive gdrive://<FOLDER_ID>"
echo "    dvc push"
echo
echo "FOLDER_ID is the trailing part of the Drive folder's URL. The first"
echo "push opens a browser for Google sign-in. Share that folder with each"
echo "teammate by address — a link alone is not enough — and commit both"
echo ".dvc/config and the pointers, which .gitignore already allows."
echo
echo "Push the CSVs, not the images: the API's rate limits make 84,916 small"
echo "files slow and failure-prone, and the default OAuth client's quota is"
echo "shared with every other DVC user."
