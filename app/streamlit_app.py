"""
streamlit_app.py — front end and live presentation for the Rakuten MLOps project.

Doubles as the defence material: the brief allows presenting from the
Streamlit app instead of slides, so the first page carries the architecture
and the findings, and the rest is a working demo of the running system.

It talks to the API over HTTP only — no TensorFlow, no model loading. That
keeps this container small and means the page cannot accidentally serve a
different model than the API does.

    streamlit run app/streamlit_app.py
"""

import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("RAKUTEN_API_URL", "http://127.0.0.1:8000").rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://127.0.0.1:8080")

st.set_page_config(page_title="Rakuten MLOps", page_icon="📦", layout="wide")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_get(path, timeout=10):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def api_post(path, payload, timeout=120):
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def render_api_status():
    health, error = api_get("/health", timeout=5)
    if error:
        st.sidebar.error(f"API unreachable\n\n{API_URL}")
        return None

    ready = health.get("model_loaded")
    # Written as a statement, not a ternary expression: Streamlit's "magic"
    # turns any bare top-level expression into st.write(), so the value
    # returned here would be rendered as a dataframe and raise.
    if ready:
        st.sidebar.success("API online")
    else:
        st.sidebar.warning("API degraded")
    st.sidebar.caption(
        f"model v{health.get('model_version')} · {health.get('model_source')}"
    )
    return health


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_architecture():
    st.title("Rakuten product classification — MLOps pipeline")
    st.caption(
        "Multimodal classification of e-commerce products into 27 Rakuten "
        "product type codes, from the product title, its description and its "
        "image."
    )

    st.subheader("Architecture")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#f4f4f5",
                  fontname="Helvetica", fontsize=10];

            subgraph cluster_data {
                label="Data"; style=dashed; color="#a1a1aa";
                csv [label="Rakuten CSVs\\n+ 84,916 images"];
                sqlite [label="SQLite\\nproducts / predictions\\n/ drift_reports"];
            }

            subgraph cluster_train {
                label="Training (batch)"; style=dashed; color="#a1a1aa";
                training [label="training.py\\nLSTM + VGG16 + blend"];
                mlflow [label="MLflow\\ntracking + Model Registry"];
            }

            subgraph cluster_serve {
                label="Serving (real time)"; style=dashed; color="#a1a1aa";
                api [label="FastAPI\\n/predict/ /training/ /metrics"];
                ui [label="Streamlit"];
            }

            subgraph cluster_monitor {
                label="Monitoring"; style=dashed; color="#a1a1aa";
                drift [label="Evidently drift service\\nPOST /run"];
                prom [label="Prometheus"];
                grafana [label="Grafana\\n2 dashboards"];
            }

            subgraph cluster_orch {
                label="Orchestration"; style=dashed; color="#a1a1aa";
                airflow [label="Airflow\\nrakuten_training\\nrakuten_drift_check",
                         fillcolor="#e0e7ff"];
            }

            csv -> sqlite [label="one-time import"];
            sqlite -> training;
            training -> mlflow [label="log + register"];
            mlflow -> api [label="champion"];
            ui -> api;
            api -> sqlite [label="store predictions"];
            sqlite -> drift;
            drift -> mlflow [label="report"];
            drift -> sqlite [label="summary"];
            api -> prom [label="/metrics"];
            prom -> grafana;
            airflow -> drift [label="hourly check"];
            airflow -> api [label="train + reload"];
        }
        """
    )

    st.subheader("How a model reaches production")
    st.markdown(
        """
        1. `training.py` trains the text and image branches and searches the
           blend weights.
        2. Every run is logged to MLflow with its parameters, metrics and a
           self-contained artifact bundle.
        3. The run is registered as a new **model version**, then scored
           against the reigning champion on held-out **weighted F1**.
        4. It is promoted to `champion` only if it wins; otherwise it stays a
           `challenger` and production is untouched.
        5. The API downloads the champion at startup. What is served is
           whatever the registry blessed — not whatever happens to be on disk.
        """
    )

    st.subheader("What we found in the baseline")
    st.markdown(
        """
        The starting repository contained four mismatches between how the
        model was **trained** and how it was **served**. None of them raised
        an error; all of them silently degraded predictions.
        """
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Mismatch": "Blend weights",
                    "Training wrote": "models/best_weights.pkl",
                    "Serving read": "models/best_weights.json",
                    "Effect": "Retraining never changed a single prediction",
                },
                {
                    "Mismatch": "Class mapping",
                    "Training wrote": "models/mapper.pkl",
                    "Serving read": "models/mapper.json",
                    "Effect": "Classes decoded with a stale mapping",
                },
                {
                    "Mismatch": "Image preprocessing",
                    "Training wrote": "raw 0-255 pixels",
                    "Serving read": "vgg16.preprocess_input",
                    "Effect": "Image branch unusable, blend weight 0.0",
                },
                {
                    "Mismatch": "Text assembly",
                    "Training wrote": "designation + description",
                    "Serving read": "description only",
                    "Effect": "Title-only products scored as empty strings",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        "All four are now covered by tests that fail in CI if the training "
        "and serving paths diverge again.",
        icon="🧪",
    )


def page_predict():
    st.title("Live prediction")

    health = api_get("/health", timeout=5)[0]
    if not health:
        st.error(f"The API at {API_URL} is not reachable.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        dataset_path = st.text_input(
            "Dataset (CSV)", value="data/preprocessed/X_test_update.csv"
        )
        images_path = st.text_input(
            "Images directory", value="data/preprocessed/image_test"
        )
    with col2:
        limit = st.number_input("Rows to score", min_value=1, max_value=200, value=10)
        st.caption(
            "Each row runs a VGG16 forward pass on CPU, so large batches are "
            "slow by design rather than by accident."
        )

    if st.button("Run prediction", type="primary"):
        with st.spinner("Scoring…"):
            result, error = api_post(
                "/predict/",
                {
                    "dataset_path": dataset_path,
                    "images_path": images_path,
                    "limit": int(limit),
                },
            )

        if error:
            st.error(error)
            return

        st.success(
            f"{result['count']} predictions · model v{result['model_version']} "
            f"({result['model_source']})"
        )
        frame = pd.DataFrame(result["predictions"])
        st.dataframe(
            frame[["designation", "prdtypecode", "confidence"]],
            hide_index=True,
            use_container_width=True,
        )

        st.caption("Predicted class distribution")
        st.bar_chart(frame["prdtypecode"].value_counts())

        with st.expander("Full response"):
            st.json(result)


def page_model():
    st.title("Model and registry")

    info, error = api_get("/model-info")
    if error:
        st.error(error)
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Serving version", info.get("model_version") or "local")
    col2.metric("Source", info.get("model_source", "—"))
    col3.metric("Predictions stored", info.get("predictions_stored") or 0)
    st.caption(f"Loaded at {info.get('loaded_at')}")

    st.markdown(
        f"Run history, metric comparisons and the champion/challenger tags "
        f"live in the MLflow UI: [{MLFLOW_URL}]({MLFLOW_URL})"
    )

    # Training launched outside the service — from a developer machine, a
    # scheduled job, or a manual promotion in the MLflow UI — does not reach
    # the API on its own. This asks it to re-resolve the champion without a
    # restart.
    if st.button("Reload champion from registry"):
        result, error = api_post("/model/reload", {}, timeout=120)
        if error:
            st.error(error)
        elif result["changed"]:
            st.success(
                f"Now serving version {result['model_version']} "
                f"(was {result['previous_version'] or 'none'})."
            )
        else:
            st.info(
                f"Already serving version {result['model_version']} — the "
                "registry has no newer champion."
            )

    st.subheader("Trigger a training run")
    st.caption(
        "Returns immediately: training runs on a background thread, because a "
        "full run takes hours and an HTTP request should not be held open for "
        "that long."
    )

    with st.form("training"):
        col1, col2, col3 = st.columns(3)
        samples = col1.number_input("Samples per class", 5, 600, 15)
        epochs_lstm = col2.number_input("LSTM epochs", 1, 10, 1)
        epochs_vgg = col3.number_input("VGG16 epochs", 1, 10, 1)
        col4, col5, col6 = st.columns(3)
        val_samples = col4.number_input("Validation per class", 5, 50, 10)
        blend_samples = col5.number_input("Blend per class", 5, 50, 10)
        eval_samples = col6.number_input("Eval per class", 5, 50, 10)

        if st.form_submit_button("Start training", type="primary"):
            payload = {
                "samples_per_class": int(samples),
                "epochs_lstm": int(epochs_lstm),
                "epochs_vgg": int(epochs_vgg),
                "val_samples_per_class": int(val_samples),
                "blend_samples_per_class": int(blend_samples),
                "eval_samples_per_class": int(eval_samples),
            }
            result, error = api_post("/training/", payload, timeout=30)
            if error:
                st.error(error)
            else:
                st.success("Training started.")
                st.json(result)

    status, error = api_get("/training/status")
    if status:
        st.subheader("Last run")
        state = status.get("status")
        if state == "running":
            st.info(f"Running since {status.get('started_at')}", icon="⏳")
        elif state == "succeeded":
            st.success(f"Finished at {status.get('finished_at')}", icon="✅")
            registry = (status.get("result") or {}).get("mlflow", {})
            if registry:
                promoted = registry.get("promoted")
                st.write(
                    f"Version **{registry.get('new_version')}** — "
                    f"{'promoted to champion' if promoted else 'kept as challenger'}"
                )
                st.json(registry)
        elif state == "failed":
            st.error(status.get("error"), icon="❌")
        else:
            st.caption("No run in this session yet.")


def page_monitoring():
    st.title("Monitoring")

    info, error = api_get("/model-info")
    if error:
        st.error(error)
        return

    report = info.get("latest_drift_report")

    st.subheader("Data drift (Evidently)")
    if not report:
        st.info(
            "No drift report yet. Serve some predictions, then run:\n\n"
            "`python src/drift_detection.py --current-limit 500`",
            icon="ℹ️",
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        drifted = bool(report["dataset_drift"])
        col1.metric("Dataset drift", "DETECTED" if drifted else "none")
        col2.metric("Drifted columns", report["n_drifted_columns"])
        col3.metric("Share drifted", f"{report['share_drifted_columns']:.0%}")
        col4.metric("Rows compared", report["n_current_rows"])

        created = datetime.fromisoformat(report["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        st.caption(f"Last checked {age:.1f} h ago · report at {report['report_path']}")

        if drifted:
            st.warning(
                "Drift means the products reaching the API no longer resemble "
                "the training data, or that the model's output mix has moved. "
                "It is a symptom, not a diagnosis — live traffic has no labels, "
                "so true accuracy cannot be measured here.",
                icon="⚠️",
            )

    st.subheader("Operational metrics")
    st.markdown(
        f"""
        Prometheus scrapes `{API_URL}/metrics`. Two Grafana dashboards read
        from it — [{GRAFANA_URL}]({GRAFANA_URL}):

        * **API health** — request rate, error rate and latency per route.
        * **Model behaviour** — predicted class distribution, confidence,
          served model version, drift status and report freshness.
        """
    )

    raw, error = api_get("/metrics")
    with st.expander("Raw /metrics endpoint"):
        if error:
            st.caption("Prometheus exposition is text, not JSON — open it directly:")
        st.markdown(f"[{API_URL}/metrics]({API_URL}/metrics)")


PAGES = {
    "Overview & architecture": page_architecture,
    "Live prediction": page_predict,
    "Model & registry": page_model,
    "Monitoring": page_monitoring,
}

st.sidebar.title("Rakuten MLOps")
choice = st.sidebar.radio("", list(PAGES))
st.sidebar.divider()
render_api_status()
st.sidebar.divider()
st.sidebar.caption(f"API: {API_URL}")
st.sidebar.markdown(
    f"[MLflow]({MLFLOW_URL}) · [Airflow]({AIRFLOW_URL}) · [Grafana]({GRAFANA_URL})"
)

PAGES[choice]()
