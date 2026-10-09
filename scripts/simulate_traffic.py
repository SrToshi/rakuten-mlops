"""
simulate_traffic.py — replay held-out rows as live traffic, to demonstrate
drift detection without waiting for real production data.

The project already has everything this needs: /predict/ logs every
prediction it serves into the `predictions` table, and drift_detection.py
compares that table against a random sample of `products` (split='train').
This script is just a traffic generator for the "current" side of that
comparison — nothing new to wire up, only new rows to feed it.

Why a script instead of segmenting the dataset up front
---------------------------------------------------------
training.py only ever samples `samples_per_class` rows per class (a few
hundred, not the full ~84,000). Everything else in the `products` table,
split='train', has never been touched by any training run — it is already
a held-out pool, with no extra bookkeeping required.

Two modes
---------
  normal   — a random sample across all classes, same shape as the training
             data. Evidently should report little or no drift: this is the
             "healthy traffic" demo.
  shifted  — restricted to a handful of product categories, with the text
             truncated to a few words. Both monitored feature groups move:
             prdtypecode (prediction drift, since the output mix narrows to
             those categories) and text_length/word_count (input drift,
             since the text is deliberately short). This is the "something
             changed" demo, meant to cross RAKUTEN_DRIFT_THRESHOLD.

Usage
-----
    python scripts/simulate_traffic.py --mode normal  --batches 6 --batch-size 20
    python scripts/simulate_traffic.py --mode shifted --batches 6 --batch-size 20

Then check the result without waiting for the hourly Airflow schedule:
    curl -X POST http://localhost:8100/run

Each batch is a separate /predict/ call (dataset_path points at a fresh CSV
the script writes under data/_simulated_traffic/), spaced out with
--sleep-seconds, so Grafana's "Requests/s" and "Predictions/s" panels show a
timeline instead of one instantaneous spike.

Why data/_simulated_traffic, not a system temp dir
---------------------------------------------------
/predict/ reads dataset_path itself, inside the api container — this script
only writes the file, it never reads it back. docker-compose.yml bind-mounts
./data into /app/data, but nothing maps the host's OS temp folder into any
container. A CSV written there is invisible server-side: Predict.run() raises
FileNotFoundError, and api.py turns that into a 404 — which is exactly what
a first version of this script hit in testing, with no clue in the response
that the problem was the file's location rather than its content.
"""

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("RAKUTEN_DB_PATH", "data/rakuten.db")
API_URL = os.getenv("RAKUTEN_API_URL", "http://localhost:8000").rstrip("/")

# What the CSV needs for Predict.run(): image_path is recomputed from
# images_path + imageid + productid regardless of what (if anything) the CSV
# itself contains, so it is deliberately left out here.
CSV_COLUMNS = ["productid", "imageid", "designation", "description"]


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def list_prdtypecodes(db_path):
    """Distinct classes present in the training split, with row counts —
    useful for picking --shift-codes deliberately instead of guessing."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT prdtypecode, COUNT(*) AS n
            FROM products
            WHERE split = 'train'
            GROUP BY prdtypecode
            ORDER BY prdtypecode
            """
        ).fetchall()
    return [(row["prdtypecode"], row["n"]) for row in rows]


def _default_shift_codes(db_path, n_codes):
    """First N classes in sorted order. Deterministic, so a demo run is
    reproducible; `--list-codes` lets a presenter pick different ones."""
    codes = [code for code, _ in list_prdtypecodes(db_path)]
    if len(codes) < n_codes:
        raise ValueError(
            f"Only {len(codes)} classes exist in the training split; "
            f"cannot pick {n_codes} to shift toward"
        )
    return codes[:n_codes]


def fetch_batch(db_path, batch_size, shift_codes=None):
    """One random batch from the training split, optionally restricted to
    a subset of classes (the 'shifted' traffic)."""
    query = """
        SELECT productid, imageid, designation, description
        FROM products
        WHERE split = 'train'
    """
    params = []
    if shift_codes:
        placeholders = ",".join("?" for _ in shift_codes)
        query += f" AND prdtypecode IN ({placeholders})"
        params.extend(shift_codes)
    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(batch_size)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def shrink_text(frame, max_words):
    """Truncate to a handful of words from the title alone, dropping the
    description entirely. A short, nearly-empty product listing is what
    pulls text_length/word_count away from the training distribution."""
    frame = frame.copy()
    frame["designation"] = (
        frame["designation"]
        .fillna("")
        .apply(lambda text: " ".join(text.split()[:max_words]))
    )
    frame["description"] = ""
    return frame


def send_batch(frame, images_path, api_url, scratch_dir):
    """Write the batch to a CSV under scratch_dir and POST it to /predict/.

    scratch_dir must be a path the api container can also see — it reads
    dataset_path itself, server-side, from inside the container's own
    filesystem. A forward slash is used deliberately (not os.path.join),
    even on Windows: Windows accepts either separator when writing the file
    locally, but the API runs inside a Linux container, where a backslash
    is just a literal character, not a path separator — a Windows-style
    join here would write a file the container could never find either.
    """
    csv_path = f"{scratch_dir.rstrip('/')}/batch_{int(time.time() * 1000)}.csv"
    frame[CSV_COLUMNS].to_csv(csv_path, index=False)

    response = requests.post(
        f"{api_url}/predict/",
        json={
            "dataset_path": csv_path,
            "images_path": images_path,
            "limit": len(frame),
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def run(args):
    if args.list_codes:
        for code, n in list_prdtypecodes(args.db_path):
            print(f"{code}\t{n} rows")
        return 0

    shift_codes = None
    if args.mode == "shifted":
        shift_codes = (
            args.shift_codes.split(",")
            if args.shift_codes
            else _default_shift_codes(args.db_path, args.n_shift_codes)
        )
        logger.info("Shifted mode — restricting traffic to classes: %s", shift_codes)

    if args.batch_size > args.max_predict_rows:
        raise SystemExit(
            f"--batch-size {args.batch_size} exceeds RAKUTEN_MAX_PREDICT_ROWS "
            f"({args.max_predict_rows}); the API will reject it with a 422"
        )

    # Deliberately NOT tempfile.TemporaryDirectory(): that lives under the
    # OS temp folder, which docker-compose never mounts into the api
    # container. This has to be somewhere the container can also read —
    # i.e. under data/, which docker-compose.yml bind-mounts into
    # /app/data. Wiped and recreated each run so old batches don't pile up;
    # /data/ is already gitignored, so nothing here touches version control.
    scratch_dir = args.scratch_dir
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir, exist_ok=True)

    for i in range(1, args.batches + 1):
        frame = fetch_batch(args.db_path, args.batch_size, shift_codes)
        if frame.empty:
            logger.warning("Batch %d/%d: no rows matched — skipping", i, args.batches)
            continue

        if args.mode == "shifted":
            frame = shrink_text(frame, args.max_words)

        result = send_batch(frame, args.images_path, args.api_url, scratch_dir)
        predictions = result.get("predictions", result)
        n = len(predictions) if isinstance(predictions, (list, dict)) else "?"
        logger.info(
            "Batch %d/%d (%s): sent %d rows, API returned %s predictions",
            i, args.batches, args.mode, len(frame), n,
        )

        if i < args.batches:
            time.sleep(args.sleep_seconds)

    logger.info(
        "Done. Check the result now, without waiting for the hourly schedule:\n"
        "    curl -X POST %s/run",
        args.drift_url,
    )
    return 0


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["normal", "shifted"], default="normal")
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=30)
    parser.add_argument("--max-words", type=int, default=3,
                         help="shifted mode only: words kept from the title")
    parser.add_argument("--shift-codes", type=str, default=None,
                         help="comma-separated prdtypecodes to restrict shifted "
                              "traffic to; default picks the first N automatically")
    parser.add_argument("--n-shift-codes", type=int, default=3)
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    parser.add_argument(
        "--scratch-dir", type=str, default="data/_simulated_traffic",
        help="where batch CSVs are written before each /predict/ call — must "
             "be inside the data/ folder docker-compose mounts into the api "
             "container; an OS temp dir is invisible to it and causes a 404"
    )
    parser.add_argument("--images-path", type=str,
                         default="data/preprocessed/image_train")
    parser.add_argument("--api-url", type=str, default=API_URL)
    parser.add_argument("--drift-url", type=str,
                         default=os.getenv("RAKUTEN_DRIFT_URL", "http://localhost:8100"))
    parser.add_argument("--max-predict-rows", type=int,
                         default=int(os.getenv("RAKUTEN_MAX_PREDICT_ROWS", "200")))
    parser.add_argument("--list-codes", action="store_true",
                         help="print distinct prdtypecodes with row counts and exit")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
