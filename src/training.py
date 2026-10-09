"""
training.py — MLOps entrypoint to (re)train the Rakuten classification model.

Wraps the whole training pipeline inside run_training(): importable without
side effects, and explicitly invokable (via CLI, via a scheduled job, or by
the /training/ endpoint of the API).

What this adds on top of a plain training script:
  * every run is tracked in MLflow (params, metrics, artifacts)
  * the resulting bundle is registered as a new Model Registry version
  * the new version is compared against the reigning champion and promoted
    only if it wins on the primary metric (see mlflow_utils.promote_if_better)
  * MLflow failures never abort training — a dead tracking server must not
    cost us a multi-hour run.

Run it directly for a quick smoke test:
    python src/training.py --epochs-lstm 1 --epochs-vgg 1 --samples-per-class 60
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile

import tensorflow as tf
from tensorflow import keras

from features.build_features import DataImporter, ImagePreprocessor, TextPreprocessor
from models.train_model import ImageVGG16Model, TextLSTMModel, concatenate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"

# The exact set of files a prediction needs. Everything here is copied into
# the MLflow artifact bundle, so a registered version is self-contained and
# the API can serve it without reaching back into this working directory.
SERVING_ARTIFACTS = [
    "best_lstm_model.h5",
    "best_vgg16_model.h5",
    "concatenate.h5",
    "tokenizer_config.json",
    "best_weights.json",
    "mapper.json",
]


def _clamp(name, requested, labels):
    """Limit a per-class sample size to what the split actually contains.

    concatenate.predict() samples without replacement, so asking for more
    rows than a class holds raises. Clamping with a warning keeps a small
    smoke run working instead of failing on an arithmetic detail.
    """
    available = int(labels.value_counts().min()) if len(labels) else 0
    if requested > available:
        logger.warning(
            "%s=%d exceeds the %d rows per class available; using %d",
            name,
            requested,
            available,
            available,
        )
        return available
    return requested


def _write_best_weights(best_weights):
    """Persist the ensemble blend weights in BOTH formats.

    Historically training wrote only best_weights.pkl while predict.py and
    api.py read best_weights.json — so a retrained model never changed a
    single prediction. The JSON is the one that is actually consumed; the
    pickle is kept for backwards compatibility with the original scripts.
    """
    weights = [float(best_weights[0]), float(best_weights[1])]

    with open(f"{MODELS_DIR}/best_weights.pkl", "wb") as fh:
        pickle.dump(tuple(weights), fh)

    with open(f"{MODELS_DIR}/best_weights.json", "w", encoding="utf-8") as fh:
        json.dump(weights, fh)

    return weights


def _collect_metrics(history, prefix):
    """Flatten a Keras history into MLflow-friendly scalar metrics."""
    if not history:
        return {}
    return {
        f"{prefix}_{key}": float(values[-1])
        for key, values in history.items()
        if values
    }


def run_training(
    epochs_lstm: int = 1,
    epochs_vgg: int = 1,
    samples_per_class: int = 600,
    batch_size: int = 32,
    val_samples_per_class: int = 50,
    blend_samples_per_class: int = 50,
    eval_samples_per_class: int = 20,
    test_samples_per_class: int = None,
    experiment_name: str = None,
    register: bool = True,
) -> dict:
    """Train the LSTM + VGG16 fusion model and track the run in MLflow.

    samples_per_class drives how much of the dataset is used (600 is the
    original full run; drop it to ~60 for a fast end-to-end smoke test).
    """
    params = {
        "epochs_lstm": epochs_lstm,
        "epochs_vgg": epochs_vgg,
        "samples_per_class": samples_per_class,
        "batch_size": batch_size,
        "val_samples_per_class": val_samples_per_class,
        "blend_samples_per_class": blend_samples_per_class,
        "eval_samples_per_class": eval_samples_per_class,
        "test_samples_per_class": test_samples_per_class,
    }
    logger.info("Starting training run with %s", params)

    # --- MLflow session (optional: training must survive a dead server) ---
    run_ctx, mlflow, mlflow_utils = _start_mlflow_run(experiment_name, params)

    try:
        # --- data -------------------------------------------------------
        data_importer = DataImporter()
        df = data_importer.load_data()

        # The test set is capped at the number of rows that will actually be
        # scored. Preprocessing is not free — every row is lemmatised with
        # NLTK — so there is no point preparing tens of thousands of rows the
        # evaluation would then subsample away.
        if test_samples_per_class is None:
            test_samples_per_class = eval_samples_per_class

        X_train, X_val, X_test, y_train, y_val, y_test = (
            data_importer.split_train_test(
                df,
                samples_per_class=samples_per_class,
                val_samples_per_class=val_samples_per_class,
                test_samples_per_class=test_samples_per_class,
            )
        )

        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor()
        for frame in (X_train, X_val, X_test):
            text_preprocessor.preprocess_text_in_df(frame, columns=["description"])
            image_preprocessor.preprocess_images_in_df(frame)

        metrics = {
            "train_rows": float(len(X_train)),
            "val_rows": float(len(X_val)),
            "test_rows": float(len(X_test)),
        }

        # --- text branch ------------------------------------------------
        logger.info("Training LSTM model")
        text_lstm_model = TextLSTMModel()
        lstm_history = text_lstm_model.preprocess_and_fit(
            X_train, y_train, X_val, y_val, epochs=epochs_lstm, batch_size=batch_size
        )
        metrics.update(_collect_metrics(lstm_history, "lstm"))
        logger.info("Finished training LSTM")

        # --- image branch -----------------------------------------------
        logger.info("Training VGG16 model")
        image_vgg16_model = ImageVGG16Model()
        vgg_history = image_vgg16_model.preprocess_and_fit(
            X_train, y_train, X_val, y_val, epochs=epochs_vgg, batch_size=batch_size
        )
        metrics.update(_collect_metrics(vgg_history, "vgg16"))
        logger.info("Finished training VGG16")

        # --- fusion -----------------------------------------------------
        with open(f"{MODELS_DIR}/tokenizer_config.json", "r", encoding="utf-8") as fh:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(fh.read())
        lstm = keras.models.load_model(f"{MODELS_DIR}/best_lstm_model.h5")
        vgg16 = keras.models.load_model(f"{MODELS_DIR}/best_vgg16_model.h5")

        # The blend weights are a hyperparameter, so they are searched on
        # VALIDATION. Searching them on training data — as this pipeline
        # originally did — made the text branch look better than it is,
        # because it memorises the training rows more readily than the image
        # branch does, and the search handed it all the weight.
        model_concatenate = concatenate(tokenizer, lstm, vgg16)

        blend_n = _clamp("blend_samples_per_class", blend_samples_per_class, y_val)
        logger.info("Searching the blend weights on validation data")
        lstm_proba, vgg16_proba, y_blend = model_concatenate.predict(
            X_val, y_val, new_samples_per_class=blend_n
        )
        best_weights, blend_accuracy = model_concatenate.optimize(
            lstm_proba, vgg16_proba, y_blend
        )
        metrics["ensemble_blend_accuracy"] = float(blend_accuracy)
        logger.info(
            "Best weights: %s (validation accuracy %.4f)", best_weights, blend_accuracy
        )

        # TEST is scored last and only once. It is the only number promotion
        # depends on, and no fitting decision — not the networks' early
        # stopping, not the blend search — has seen these rows.
        eval_n = _clamp("eval_samples_per_class", eval_samples_per_class, y_test)
        logger.info("Scoring the ensemble on the held-out test set")
        test_scores = model_concatenate.evaluate(
            X_test, y_test, best_weights, samples_per_class=eval_n
        )
        metrics["ensemble_test_accuracy"] = test_scores["accuracy"]
        metrics["ensemble_test_weighted_f1"] = test_scores["weighted_f1"]
        metrics["ensemble_test_rows"] = float(test_scores["n_samples"])
        logger.info("Test: %s", test_scores)

        # --- persist artifacts ------------------------------------------
        weights = _write_best_weights(best_weights)

        num_classes = 27
        proba_lstm = keras.layers.Input(shape=(num_classes,))
        proba_vgg16 = keras.layers.Input(shape=(num_classes,))
        weighted_proba = keras.layers.Lambda(
            lambda x: weights[0] * x[0] + weights[1] * x[1]
        )([proba_lstm, proba_vgg16])
        concatenate_model = keras.models.Model(
            inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
        )
        concatenate_model.save(f"{MODELS_DIR}/concatenate.h5")

        result = {
            "status": "ok",
            "best_weights": weights,
            "params": params,
            "metrics": metrics,
        }

        # --- track + register -------------------------------------------
        if run_ctx is not None:
            _log_metrics(mlflow, metrics)
            registry_info = _log_and_register(
                mlflow, mlflow_utils, register=register
            )
            result["mlflow"] = registry_info

        return result

    finally:
        if run_ctx is not None:
            try:
                mlflow.end_run()
            except Exception:  # noqa: BLE001 - never mask a training error
                pass


# ---------------------------------------------------------------------------
# MLflow plumbing, kept apart so the training flow above stays readable and so
# every failure mode here is non-fatal.
# ---------------------------------------------------------------------------


def _start_mlflow_run(experiment_name, params):
    """Open an MLflow run. Returns (run, mlflow_module, mlflow_utils_module).

    Returns (None, None, None) when MLflow is unavailable or misconfigured,
    in which case training proceeds untracked rather than failing.
    """
    if os.getenv("MLFLOW_DISABLED", "").lower() in ("1", "true", "yes"):
        logger.info("MLflow disabled via MLFLOW_DISABLED")
        return None, None, None

    # Starting the run is the part that may legitimately fail (no server, bad
    # URI, no credentials). Only this is allowed to turn tracking off.
    try:
        import mlflow

        import mlflow_utils

        mlflow_utils.setup_experiment(
            experiment_name or mlflow_utils.DEFAULT_EXPERIMENT
        )
        run = mlflow.start_run()
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow unavailable (%s) — training continues untracked", exc)
        return None, None, None

    logger.info(
        "MLflow run %s at %s", run.info.run_id, mlflow_utils.get_tracking_uri()
    )

    # Everything below enriches the run. Each piece fails on its own and is
    # reported as itself: an earlier version wrapped the whole block in one
    # try, so a missing data_version module was reported as "MLflow
    # unavailable" and silently cost a five-hour run its tracking — the run
    # was created, then left orphaned with no params, metrics or model.
    try:
        mlflow.log_params(params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not log parameters: %s", exc)

    _tag_data_version(mlflow)

    return run, mlflow, mlflow_utils


def _tag_data_version(mlflow):
    """Tag the run with the DVC hashes of the data it used.

    Optional by design: DVC may not be set up, and that must not cost the run
    its tracking.
    """
    try:
        import data_version

        tags = data_version.mlflow_tags()
        if tags:
            mlflow.set_tags(tags)
            logger.info("Data version recorded: %s", tags)
        else:
            logger.info("No DVC pointers found — data version not recorded")
    except ImportError:
        logger.warning(
            "src/data_version.py is missing — this run will not record which "
            "data version produced it"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not record the data version: %s", exc)


def _log_metrics(mlflow, metrics):
    try:
        mlflow.log_metrics(metrics)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not log metrics to MLflow: %s", exc)


def _log_and_register(mlflow, mlflow_utils, register=True):
    """Upload the serving bundle and register it as a new model version."""
    info = {}
    try:
        run_id = mlflow.active_run().info.run_id

        # Copy exactly the files a prediction needs into a clean directory,
        # so the artifact bundle is deterministic rather than "whatever is
        # sitting in models/ right now".
        with tempfile.TemporaryDirectory() as bundle:
            for name in SERVING_ARTIFACTS:
                src = os.path.join(MODELS_DIR, name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(bundle, name))
                else:
                    logger.warning("Missing artifact, not bundled: %s", name)
            mlflow.log_artifacts(bundle, artifact_path="model")

        info["run_id"] = run_id

        if register:
            version = mlflow_utils.register_model_version(run_id, "model")
            info.update(mlflow_utils.promote_if_better(version))
            logger.info("Registry decision: %s", info)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow artifact/registry step failed: %s", exc)
        info["error"] = str(exc)
    return info


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the Rakuten fusion model")
    parser.add_argument("--epochs-lstm", type=int, default=1)
    parser.add_argument("--epochs-vgg", type=int, default=1)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=600,
        help="Training rows per class. Use ~60 for a fast smoke test.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--val-samples-per-class",
        type=int,
        default=50,
        help="Validation rows per class used during fitting. Each one costs a "
        "VGG16 forward pass per epoch, so this dominates short runs.",
    )
    parser.add_argument(
        "--blend-samples-per-class",
        type=int,
        default=50,
        help="Validation rows per class used to search the blend weights.",
    )
    parser.add_argument(
        "--eval-samples-per-class",
        type=int,
        default=20,
        help="Test rows per class used to score the ensemble. This is the "
        "number the champion/challenger decision rests on.",
    )
    parser.add_argument(
        "--test-samples-per-class",
        type=int,
        default=None,
        help="Size of the held-out test split. Defaults to "
        "--eval-samples-per-class, since only that many rows get scored.",
    )
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Track the run but do not create a Model Registry version.",
    )
    return parser.parse_args()


def _report_tracking_destination():
    """Say where this run will be recorded, before it starts.

    run_training() works with or without a tracking server, and which one it
    uses is decided by an environment variable rather than by a flag — so
    from a terminal there is nothing to see. The two destinations are not
    interchangeable: the file store belongs to this checkout, while every
    service in the stack reads the tracking server. A run sent to the wrong
    one finishes successfully, registers a model version, prints its metrics,
    and is then invisible to the API. A failure with no error attached to it
    is the expensive kind, so this prints the destination either way.

    It does not refuse to run. Training against the file store is the
    supported way to experiment without standing up a server, and it is why
    the default exists (see mlflow_utils.get_tracking_uri).
    """
    import mlflow_utils

    print(f"[training] MLflow tracking: {mlflow_utils.get_tracking_uri()}")
    if os.getenv("MLFLOW_TRACKING_URI"):
        return

    print(
        "[training] MLFLOW_TRACKING_URI is not set, so this run goes to a local\n"
        "[training] file store. The stack's services read the tracking server, so\n"
        "[training] they will not see the model this run registers.\n"
        "[training]\n"
        "[training] To record it where they can:\n"
        "[training]     set MLFLOW_TRACKING_URI=http://127.0.0.1:5000     (Windows)\n"
        "[training]     export MLFLOW_TRACKING_URI=http://127.0.0.1:5000  (Linux/macOS)\n"
        "[training]\n"
        "[training] Or train through the API, which is already configured and\n"
        "[training] reloads the champion when the run finishes:\n"
        "[training]     curl -X POST http://127.0.0.1:8000/training/",
        file=sys.stderr,
    )


if __name__ == "__main__":
    args = _parse_args()
    _report_tracking_destination()
    outcome = run_training(
        epochs_lstm=args.epochs_lstm,
        epochs_vgg=args.epochs_vgg,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        val_samples_per_class=args.val_samples_per_class,
        blend_samples_per_class=args.blend_samples_per_class,
        eval_samples_per_class=args.eval_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        experiment_name=args.experiment,
        register=not args.no_register,
    )
    print(json.dumps(outcome, indent=2, default=str))
