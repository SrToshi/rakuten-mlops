# Rakuten — multimodal product classification
## Project guide: objectives, decisions and demo

*Spanish version: `docs/guia-proyecto-es.md` — identical content.*

---

## How to read this guide

It is written for someone who knows MLOps concepts from a distance and wants
to understand **what each tool does in general** and, separately, **what it
does in this project**. Those are two different things and they do not always
match: Airflow can do far more than it does here, and that gap is itself a
decision worth being able to defend.

Part 1 sets up the problem. Part 2 holds the strategic decisions, which is
where the marks are. Part 3 explains the tools one by one. Part 4 walks the
whole system twice, following a concrete piece of data. Parts 5 and 6 are
operational: the demo commands and what every panel means. Part 7 anticipates
questions and Part 8 lists what the system does **not** do.

If you only have ten minutes before the defense: Part 2, Part 5, Part 6.

---

# Part 1 — The project

## 1.1 The problem

Rakuten is a marketplace. When a seller lists a product, someone has to decide
which category it belongs to. The catalogue has **27 categories** (the
`prdtypecode` values), and sorting tens of thousands of products by hand does
not scale.

Each product has three parts:

- **designation** — the title, always present.
- **description** — free text, frequently empty.
- **image** — a photo of the product; there are 84,916 in the training set.

This is a **multimodal** problem: the answer is split between the text and the
image, and neither alone is enough. A cable and a charger look alike in a
photo and differ in the title; two books with generic titles differ by their
cover.

## 1.2 What the brief asked for

The exercise is not "train the best model". It is **put a model into
production and keep it alive**. The objectives, as stated:

1. **Collect the data and store it in a SQL (or NoSQL) database locally**,
   with a Python script that runs once.
2. **Version the data** with DVC, without Git, and store its hashes in MLflow.
3. **Serve the model** behind an API.
4. **Track training runs** in MLflow: parameters, metrics and artifacts.
5. **Compare versions** and promote the better one to production.
6. **Monitor** the service and detect *drift* in the data.
7. **Build dashboards** in Grafana: one for data drift, one for API health.
8. **Orchestrate** the cycle with Airflow.
9. **Containerise** everything, with CI.

Note what it does **not** ask for: maximum accuracy. That changes which model
is the right one, and it is the first strategic decision.

## 1.3 The system at a glance

Seven services, each in its own container:

```
                      ┌──────────────────────────────────────┐
    Data              │              Monitoring              │
  ┌──────────────┐    │  ┌─────────────┐    ┌─────────────┐  │
  │ Rakuten CSVs │    │  │  Evidently  │    │ Prometheus  │  │
  │ + 84,916 img │    │  │ POST /run   │    │   :9090     │  │
  └──────┬───────┘    │  │   :8100     │    └──────┬──────┘  │
         │ one-time   │  └──────┬──────┘           │         │
         │  import    │         │                  ▼         │
         ▼            │         │           ┌─────────────┐  │
  ┌──────────────┐    │         │           │   Grafana   │  │
  │    SQLite    │◄───┼─────────┘           │ 2 dashboards│  │
  │  products    │    │  summary            │    :3000    │  │
  │  predictions │    └─────────────────────┴──────┬──────┘  │
  │  drift_reports│                                │         │
  └──────┬───────┘                                 └─────────┘
         │  ▲                     Orchestration
         │  │ stores every   ┌──────────────────────┐
         │  │ prediction     │       Airflow        │
         │  │                │  rakuten_training    │
         │  │                │  rakuten_drift_check │
         │  │                │        :8080         │
         │  │                └───┬──────────────┬───┘
         │  │        hourly ─────┘              │ train + reload
         │  │        POST /run                  ▼
         ▼  │                            ┌──────────────┐
  ┌──────────────┐     Serving           │   FastAPI    │
  │ training.py  │  ┌─────────────┐      │  /predict/   │
  │ LSTM + VGG16 │  │  Streamlit  │─────▶│  /training/  │
  │   + blend    │  │    :8501    │      │  /metrics    │
  └──────┬───────┘  └─────────────┘      │    :8000     │
         │                               └──────▲───────┘
         │ registers     ┌──────────────┐       │
         └──────────────▶│    MLflow    │───────┘
                         │  tracking +  │ champion
                         │   Registry   │
                         │    :5000     │
                         └──────────────┘
```

| Service | Port | Role |
|---|---|---|
| `api` | 8000 | FastAPI: serves predictions, starts training, exposes metrics |
| `streamlit` | 8501 | Demonstration interface |
| `mlflow` | 5000 | Experiment tracking and Model Registry |
| `drift` | 8100 | Evidently service; computes drift on request |
| `airflow` | 8080 | Orchestrator: two DAGs |
| `prometheus` | 9090 | Scrapes the API's metrics |
| `grafana` | 3000 | Dashboards and alerts |

## 1.4 The model, in two paragraphs

Two branches, trained separately and combined at the end:

- **Text branch** — an LSTM network over `designation + description`. The text
  is tokenised (each word becomes a number according to a learned vocabulary)
  and the LSTM reads that sequence in order.
- **Image branch** — **VGG16**, a convolutional network pre-trained on
  ImageNet. What it already knows how to see (edges, textures, shapes) is
  reused, and only the final layer is trained for the 27 categories. This is
  *transfer learning*, and it is what makes training on 84,916 images possible
  instead of the millions it would take from scratch.

Each branch produces 27 probabilities. The final prediction is a **weighted
average** of the two: `w_text × P_text + w_image × P_image`. Those two weights
are not chosen by hand; they are searched over a set held out for that
purpose. The result weighs about 150 MB and answers in well under a second on
CPU.

---

# Part 2 — The strategic decisions

This is what separates a passing project from a good one. Each decision is
presented with what it costs, not only what it gains: a decision with no
acknowledged cost is usually a decision that has not been thought through.

## 2.1 Seven containers instead of one

**The decision.** Each service runs in its own image, with its own Python
environment, and they talk over HTTP and a shared volume.

**Why.** This is not architectural decoration: they genuinely **do not fit
together**.

- TensorFlow 2.13 pins `typing_extensions < 4.6` and `protobuf < 5`.
- The MLflow server needs SQLAlchemy and Alembic, which require
  `typing_extensions >= 4.6`.
- Evidently brings Litestar and its own Uvicorn extras.

These are mutually incompatible requirements. A single Python environment with
all three simply will not install. Isolating each dependency set in its own
image is what makes the stack **installable**, not what makes it elegant.

**What it costs.** Seven containers on a two-core machine compete for CPU. Two
failures during this project are a direct consequence: the Airflow webserver
died on a 120 s gunicorn timeout during its first boot, and the MLflow
artifact download failed at exactly 30 s on a different gunicorn's default
timeout. Both were fixed by raising the timeouts to 300 s. The lesson: when
little CPU is shared among many processes, failures appear as timeouts, not as
logic errors.

The lesson had a second instalment. With a retrain running inside the `api`
container, 300 s was **not enough either**: the scheduler logged `Heartbeat
recovered after 41.33 seconds` and the webserver exited with `No response
from gunicorn master within 300 seconds`. Not an Airflow bug and not a memory
problem (`OOMKilled` was false) — the supervisor gave up waiting on a master
process that was merely queuing for CPU. It is now 900 s and a single worker,
because with the `SequentialExecutor` and one viewer the second worker bought
nothing and doubled the boot work on the scarcest resource.

As a safety net, all seven services carry `restart: unless-stopped`. A
container that dies on a load spike comes back on its own instead of staying
down until somebody looks at `docker compose ps`. Which is exactly how this
failure surfaced: Airflow had been stopped for four hours with nothing
saying so.

**And the root cause of Airflow's slow boot was something else.** Its
metadata database is SQLite and it lived inside the container rather than on
a volume. Every `docker compose up --build` recreates the container and
destroys it, so Airflow re-initialised *everything* on each build:
migrations, several hundred Flask-AppBuilder permission rows (the `Added
Permission ...` wall in the logs) and — the part that hurts in a demo — **the
DAG run history and the DAGs' paused state**. Those boot minutes were not
inherent to Airflow: they were rebuilding it from scratch each time.

The database now lives on a named volume (`airflow-db`), pointed at by
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`. The directory is created in the
Dockerfile with the right owner, because Docker seeds an empty named volume
from the image path including its ownership; created at mount time it would
belong to `root` and Airflow could not write to it. The first boot still pays
the initialisation once; later ones do not.

## 2.2 The light model instead of the most accurate one

**The decision.** The LSTM + VGG16 model (~150 MB) is served, not a teammate's
ensemble scoring 0.9151 weighted F1.

**Why.** The two models are judged on different axes. The teammate's is seven
fine-tuned encoders, roughly 4.5 GB of weights, several seconds per prediction
on CPU. For a project assessed on MLOps criteria — image size, versioning
cost, serving latency, a demo that has to work live — the lighter model is the
defensible choice.

**What it costs.** Accuracy. Say so plainly when asked.

**The nuance that makes it a good answer**: the right home for the stronger
model is the **Model Registry, as a competing version**. That is exactly the
comparison a registry exists to make. The model has not been discarded; it has
been put where the system knows how to evaluate it.

## 2.3 The Model Registry as the single source of truth

**The decision.** What the API serves is what the registry blessed, not a file
someone left on disk.

**Why.** It is the difference between "we have a model" and "we know which
model we have". Every training run is registered as a **new version**, scored
against the reigning champion on weighted F1 over held-out data, and
**promoted only if it wins**. If it loses, it stays a *challenger* and
production is untouched.

```
train ──▶ register version ──▶ does it beat the champion on
                                ensemble_test_weighted_f1?
                                     │
                        ┌────────────┴────────────┐
                       yes                        no
                        │                          │
                   tag: champion            tag: challenger
                   the API downloads it     production untouched
```

**What it costs.** A network dependency at API startup. If MLflow does not
answer, something has to happen — which leads to the next decision.

## 2.4 Degradation is deliberate

**The decision.** If MLflow is down, the API **still starts**, loads the model
from the local directory and says so: `model_source: local-directory` in
`/health`. Training also continues, untracked.

**Why.** A dead tracking server must not cost a multi-hour run, nor take the
prediction service down with it. Observability should degrade; it should not
take production with it.

**The detail that makes it honest**: the degraded path is **labelled
differently**. The Grafana panel shows "local fallback" instead of a version
number. That is what caught a real failure during development — the API had
been serving from disk for hours without anyone knowing, because the champion
download was failing on the gunicorn timeout. A silent fallback would have
hidden the problem; a labelled one exposed it.

## 2.5 Airflow orchestrates and computes nothing

**The decision.** The DAGs import neither TensorFlow, nor Evidently, nor the
training code. Every task is an **HTTP call** to a service.

**Why.** If Airflow imported the training code, TensorFlow would have to be
installed in its image — recreating exactly the dependency conflict that
forced the stack apart (§2.1). And it would do so quietly, at build time,
weeks after anyone remembered why.

**What it costs.** Two concrete consequences, both visible in the code:

- The sensor that waits for a training run uses `mode="reschedule"`, which
  frees the execution slot between pokes. In `poke` mode it would hold the
  `SequentialExecutor`'s only slot for hours and nothing else could run.
- The drift DAG triggers the training DAG with `wait_for_completion=False`: it
  fires and moves on, so the hourly check never backs up behind a multi-hour
  run.

**This is enforced by tests.** `tests/test_dags.py` fails if a DAG imports
anything heavy, if `Dockerfile.airflow` installs TensorFlow or Evidently, or
if a DAG calls an endpoint that does not exist.

## 2.6 A single automatic retrain trigger

**The decision.** Airflow is the only automated path to retraining. Grafana's
alerts inform and do not act.

**Why.** This decision was made *during* the project, correcting an earlier
design. Grafana had a webhook pointed at `/training/` that retrained when the
drift alert fired — reasonable when there was no orchestrator. Once Airflow
arrived, it became a **second independent trigger** for the same action, on a
different schedule, firing whether or not the DAGs were unpaused.

The `/training/` endpoint serialises runs behind a lock (it answers 409 while
one is in flight), so they never collided. But that **hid** the problem rather
than solving it: when a run started, nothing recorded which of the two had
started it.

> Two automatic triggers for one action is not redundancy. It is ambiguity.

**The technical detail that matters**: Grafana's alerting provisioning is
**additive**. Removing the contact point from the YAML does not delete it from
Grafana's database, which lives in a volume that survives `docker compose
down`. An explicit delete was required:

```yaml
deleteContactPoints:
  - orgId: 1
    uid: rakuten-retrain
resetPolicies:
  - 1
```

Verified against a running instance: the contact point is gone and both alert
rules survived.

## 2.7 SQLite

**The decision.** SQLite for the project's data and for Airflow's metadata.

**Why.** The brief asks for "a SQL (or NoSQL) database locally". The wording
itself — "SQL *or* NoSQL" — says the engine does not matter; what matters is
that structured, persistent storage exists, populated by a script that runs
once. SQLite **is** a full SQL database: same language, same ACID guarantees,
without the complexity of a client-server process.

It also matches the usage profile: a single-node deployment and a database
that is read far more than it is written.

**What it costs.** No real write concurrency. The access layer is isolated in
`db.py`, so moving to Postgres is a connection-string change, not a rewrite.

**A common confusion worth heading off**: Airflow's UI warns against using
SQLite as a metadata DB in production. That refers to **its own internal
bookkeeping** (DagRuns, TaskInstances, users), which has nothing to do with
Rakuten's data. Migrating Airflow to Postgres would not touch a single row of
the `products` table.

## 2.8 SequentialExecutor

**The decision.** Airflow runs on `SequentialExecutor`: one task at a time, no
worker fleet.

**Why.** It is the right size for a single-node deployment with two DAGs and
sporadic runs. A `CeleryExecutor` with separate workers would be
over-engineering on a two-core machine already running seven containers.

**What it costs.** No parallelism at all. And it is exactly why the sensor uses
`mode="reschedule"` (§2.5): with a single slot, a sensor that occupies it
blocks the entire system.

**The sentence for the defense**: moving to `LocalExecutor` and Postgres is
*configuration, not redesign*.

## 2.9 DVC without Git

**The decision.** DVC initialised with `dvc init --no-scm`, and the data
hashes recorded as MLflow tags.

**Why.** An MLflow run already records the code's parameters and the resulting
metrics, but **not which data produced them**. Two runs with identical
parameters and different metrics are unexplainable without it. With the hash
on the run, any disagreement can be attributed to either the code or the data.

**The elegant detail**: `src/data_version.py` **reads the `.dvc` files by
hand** (they are YAML holding an md5, a size and a path) instead of importing
the `dvc` library. The training environment therefore never needs DVC
installed. It is the same pattern as with MLflow: the heavy tool lives in its
own place, and whoever consumes it only reads the artifact it produces.

**What `--no-scm` costs.** It is what the brief asks for, but it carries an
operational price worth knowing, because it is documented nowhere obvious:
**DVC writes none of the `.gitignore` files it normally would**. With Git,
`dvc init` protects the cache, the tracked data and the credentials file by
itself. Without Git it protects nothing, and three things are left exposed to
a `git add -A`: the cache (`.dvc/cache/`), the temporary files, and
`.dvc/config.local` — which holds the **remote's token in plain text**. All
three are now in the repository's `.gitignore`, written by hand.

**And a git trap that surfaces when versioning the pointers.** The `.dvc`
files live under `data/`, which was excluded wholesale. The obvious pattern
for re-including them **does not work**:

```
/data/
!data/preprocessed/*.dvc     <-- never gets a chance to apply
```

Git cannot re-include a file whose parent directory is excluded, and it fails
**silently**: no error, the files simply never appear. Each level has to be
re-included on the way down:

```
/data/*
!/data/preprocessed/
/data/preprocessed/*
!/data/preprocessed/*.dvc
```

## 2.9b The remote: why DagsHub and not Google Drive

**The decision.** The data is pushed to a DVC remote hosted on DagsHub, with
basic token authentication.

**Why not Drive, which was the obvious option.** Google **blocks DVC's
default OAuth application**. This is not a warning you can click past with
"advanced settings": it is a hard block, because the app uses restricted
scopes (full access to the user's Drive) and DVC has not managed to pass
Google's verification. Its own maintainer describes the situation as "stuck
in limbo", with no fix and no timeline. The app needs broad permissions
because it cannot know in advance which folder you will point it at.

The only route with Drive would be setting up your own Google Cloud project
with your own OAuth credentials — half an hour of work, with tokens expiring
after 7 days while the app stays in testing mode.

**Why DagsHub solves the same problem better**: token authentication rather
than OAuth, so there is no consent screen and no application a third party
can block; 100 GB free per repository; and standard endpoints that DVC treats
like any other remote.

**And inside DagsHub, the S3 endpoint rather than the HTTP one.** DagsHub
exposes the same store through two routes: `…/<repo>.dvc`, which DVC treats as
an HTTP remote, and `…/<repo>.s3`, which it treats as an S3 remote. For three
CSVs it makes no difference. For 79,200 images it makes all of it, and the
reason sits in a phase you never see: before uploading anything, DVC checks
**which objects already exist** on the remote, so it does not resend what is
already there. An HTTP remote has no bulk-listing operation, so that check
becomes one request per object — 79,200 requests before a single byte is
uploaded. The progress bar sits at `0/?`, with no denominator because it
cannot know how many remain, for hours. The S3 endpoint lists with
pagination: a few dozen calls for the same inventory.

The change is one line of configuration plus the `dvc-s3` package (which
`dvc` does not install by default: without it, `dvc push` fails with `s3 is
supported, but requires 'dvc-s3' to be installed`). And it has a trap:
`dvc remote add` over a remote that already exists rewrites the URL and
**leaves the other keys untouched** — keys that were legal for an HTTP remote
and are illegal for an S3 one. From then on every DVC command fails with
`extra keys not allowed @ data['remote']['origin']['auth']`, including the
`dvc remote remove` you would reach for to fix it, because DVC validates the
whole file before running anything. The way out is to delete
`.dvc/config.local` and rewrite it with `dvc remote modify --local`.

**The detail that separates configuration from secret.** DVC splits the
configuration across two files on purpose: `.dvc/config` holds the remote's
URL and **is versioned**, so whoever clones knows where to fetch the data
from; `.dvc/config.local` holds the credentials and **is never versioned**.
The command that writes them says so in its own name: `dvc remote modify
--local`.

## 2.10 The DAGs are born paused

**The decision.** `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: "True"`.

**Why.** Starting the stack must not launch a multi-hour training run by
surprise. Whoever wants them running unpauses them.

**What it costs.** You have to remember to unpause them before the demo. It is
on the checklist in Part 5.

---

# Part 3 — The pieces, one by one

Each tool on two levels: what it is in general, and what it does here.

## 3.1 Docker and Docker Compose

**What it is.** Docker packages an application with its whole environment —
minimal operating system, libraries, dependencies — into an **image** that
runs identically on any machine. Compose describes a set of containers and how
they relate, in a single YAML file.

**What it does here.** `docker-compose.yml` defines the seven services, their
ports, environment variables, volumes and startup dependencies. It is what
turns "install these three incompatible dependency stacks" into
`docker compose up -d`.

Details worth knowing:

- **Mounted volumes**: `./data`, `./models` and `./logs` are mounted from the
  host. ~2.4 GB of JPEGs has no business inside a Docker image. Practical
  consequence: whatever you write into `data/` on your machine, the container
  sees immediately, with no rebuild.
- **`depends_on` with `condition: service_healthy`**: the API does not start
  until MLflow answers its healthcheck.
- **What is copied vs what is mounted**: `src/` is **copied** at build time.
  So a change to `src/drift_service.py` requires
  `docker compose up -d --build drift`, while a change to a Grafana dashboard
  is picked up on its own.

## 3.2 FastAPI

**What it is.** A Python framework for building HTTP APIs. Its distinguishing
feature is that it uses Python's type annotations to validate requests
automatically and to generate interactive documentation without writing any.

**What it does here.** It is the service that serves the model. Endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/predict/` | POST | Classifies rows from a CSV; stores every prediction in SQLite |
| `/training/` | POST | Starts a background training run; returns 202 immediately |
| `/training/status` | GET | State of the run in flight |
| `/model/reload` | POST | Re-resolves the champion from the registry and loads it |
| `/health` | GET | Alive, model loaded, version and source |
| `/model-info` | GET | Detail of the model being served |
| `/metrics` | GET | Metrics in Prometheus format |

Two design decisions visible here:

- **`/training/` returns 202, not 200.** A full run takes hours; an HTTP
  request should not be held open that long. The work happens on a background
  thread and the caller polls `/training/status`.
- **`/model/reload` exists for a specific reason.** The API picks up a new
  champion automatically at startup and after a training run *it* executed.
  Neither covers a run launched outside the service — a developer running
  `training.py` directly, or a teammate promoting a version by hand in the
  MLflow UI. Without this endpoint, the only way to pick those up would be
  restarting the container, which reloads VGG16 from scratch and takes the
  service down while it does.

**Where to see it.** <http://localhost:8000/docs> — interactive documentation,
generated automatically.

## 3.3 SQLite

**What it is.** A SQL database that lives in a single file, with no server.
The library links directly into the program itself.

**What it does here.** Three tables in `data/rakuten.db`:

- **`products`** — Rakuten's data, loaded once by
  `src/data/import_to_db.py`. This is the **reference side** of drift
  detection: current traffic is compared against it.
- **`predictions`** — every prediction served, with its text, predicted class,
  confidence and model version. This is the **current side** of the
  comparison.
- **`drift_reports`** — the summary of each drift check.

The detail that explains the design: **`/predict/` stores every prediction**.
Without that there would be nothing to compare against, and drift detection
would be impossible. The write is wrapped in a deliberate `try/except`: a
failed monitoring write must never turn a correct prediction into a failed
request.

## 3.4 DVC

**What it is.** *Data Version Control*. Git versions code but does not cope
with gigabyte files. DVC solves that by storing a **pointer** in the
repository (a small `.dvc` file holding an md5 hash, a size and a path) while
the real data lives elsewhere.

**What it does here.** Initialised without Git (`dvc init --no-scm`), as the
brief specifies, so the `.dvc` files are the record themselves.
`training.py` reads those hashes through `src/data_version.py` and records
them as tags on the MLflow run, prefixed with `data.` (for example
`data.X_train_update.csv`).

What is tracked is the three CSVs and the training image directory. The
remote lives on DagsHub, so whoever clones the repository gets the pointers
with the code and retrieves the data with `dvc pull`.

**What it buys in practice.** The ability to answer "why did this run score
0.71 and that one 0.68 with the same parameters?". If the hashes match, the
difference is in the code or in randomness; if they do not, it is in the data.

**An installation detail that costs an afternoon if you don't know it.** DVC
3.55.2 declares its `pathspec` dependency **with no upper bound**, so `pip`
resolves it to `pathspec` 1.x — which dropped the private `_DIR_MARK` symbol
that DVC's ignore handling imports. The result is that **every** DVC command
fails identically, before doing any work:

```
ERROR: unexpected error - cannot import name '_DIR_MARK'
from 'pathspec.patterns.gitwildmatch'
```

Pinning `pathspec==0.12.1` fixes it.

The second case is the more instructive one, because the right fix changed
with the environment. The Drive extra drags in a 2022-era `pyOpenSSL` that
clashes with modern `cryptography` and fails with `module 'lib' has no
attribute 'GEN_EMAIL'`. The immediate answer is to pin `pyopenssl==24.2.1`,
which requires `cryptography < 44`. But installing `dvc-s3` on Python 3.14
makes pip resolve `asyncssh` to a version requiring `cryptography >= 48.0.1`,
and the two constraints have no common solution — no published `pyOpenSSL`
accepts `cryptography` 50.

The way out is not a better pin: it is that **nothing in the environment
depends on `pyOpenSSL`**. The reverse dependency tree shows it plainly — it
was there as a leftover of the Drive attempt, which was dropped. Uninstall it
and the conflict is gone. The general lesson: before hunting for the version
combination that reconciles two constraints, it is worth asking whether the
conflicting package is needed at all.

The pins and this note are in `scripts/setup_dvc.sh`.

**And DVC lives in an environment of its own**, never beside the training
one: dvc 3.x upgrades `typing_extensions` past the `< 4.6` that TensorFlow
2.13 pins. It is the same principle that separates the seven containers
(§2.1), applied to installing a local tool.

## 3.5 TensorFlow / Keras

**What it is.** The library used to define and train neural networks. Keras is
its high-level interface.

**What it does here.** Builds and trains the model's two branches, and runs
predictions. `src/training.py` orchestrates the whole process: trains the text
branch, trains the image branch, searches the blend weights, evaluates on
held-out data, and logs everything to MLflow.

**A performance note that matters for the demo**: every prediction involves a
VGG16 forward pass on CPU. `/predict/` is **slow by nature** — not a problem
to fix, but the cost of classifying images without a GPU. Say it before
someone asks.

## 3.6 MLflow

**What it is.** Two things worth not conflating:

1. **Tracking** — a lab notebook. Every run records its parameters, metrics
   and artifacts, and they become comparable with each other.
2. **Model Registry** — a catalogue of models with versions and tags, which
   answers "which model is in production right now?".

**What it does here.** Both.

- Every training run creates a run in the `rakuten-classification` experiment,
  with hyperparameters, metrics and a **self-contained artifact bundle**
  (weights, tokeniser, class mapping, blend weights).
- That run is registered as a new version of the `rakuten-fusion` model.
- `promote_if_better()` compares it with the champion on
  `ensemble_test_weighted_f1`. Only if it **strictly** wins does it get the
  `champion` tag; otherwise it stays a `challenger`.
- The API downloads the champion's artifacts at startup.

**Two operational details.**

- **`--serve-artifacts`**: the MLflow server proxies artifacts, so clients do
  not need direct access to the storage. The cost: large files pass through
  that proxy, which is why `--gunicorn-opts "--timeout 300"` was needed. With
  the 30 s default, downloading the VGG16 weights died halfway and the API
  fell back silently to the local directory.
- **`mlflow-skinny` vs the full server**: the light client can be installed
  alongside TensorFlow; the full server cannot (it is the SQLAlchemy/Alembic
  pile that starts the conflict).

**Where to see it.** <http://localhost:5000> — experiments, runs, compared
metrics, and the model registry with its tags.

## 3.7 Evidently

**What it is.** A library for detecting *drift*: the phenomenon where the data
reaching a model in production stops resembling the data it was trained on. It
compares two datasets — reference and current — column by column, with
statistical tests, and reports which have moved.

**What it does here.** It lives in its own service (`src/drift_service.py`,
port 8100) with a single working endpoint: `POST /run`.

It compares three columns:

| Column | What it compares | What a shift means |
|---|---|---|
| `text_length` | text length | incoming products have differently sized descriptions |
| `word_count` | number of words | the same, from another angle |
| `prdtypecode` | **true** labels in reference vs **predicted** in current | the output mix has moved |

**A distinction worth stating out loud before anyone asks**: the first two are
*input drift* and the third is *prediction drift*. They are different
measurements. And **ground-truth drift is not measurable here**, because live
predictions carry no label. Nobody tells us whether we were right.

The action threshold is **0.5**: if more than half the monitored columns have
drifted, a retrain is requested.

## 3.8 Prometheus

**What it is.** A time-series database that works by **scraping**: rather than
applications pushing data to it, Prometheus periodically visits an HTTP
endpoint on each application and takes whatever it finds. It stores each value
with a timestamp and lets you query it in its own language, **PromQL**.

**What it does here.** It scrapes the API's `/metrics`. Retention: 15 days.

The metrics the API exposes:

| Metric | Type | What it measures |
|---|---|---|
| `rakuten_http_requests_total` | Counter | requests, by route and status code |
| `rakuten_http_request_duration_seconds` | Histogram | latency per route |
| `rakuten_predictions_total` | Counter | predictions, by predicted class |
| `rakuten_prediction_confidence` | Histogram | confidence of each prediction |
| `rakuten_model_version` | Gauge | version served (0 = local fallback) |
| `rakuten_model_loaded` | Gauge | 1 if a model is loaded |
| `rakuten_training_in_progress` | Gauge | 1 during a training run |
| `rakuten_training_runs_total` | Counter | finished runs, by outcome |
| `rakuten_drift_detected` | Gauge | Evidently's latest verdict |
| `rakuten_drift_share_of_drifted_columns` | Gauge | share of drifted columns |
| `rakuten_drift_report_age_seconds` | Gauge | age of the latest report |
| `rakuten_predictions_stored_total` | Gauge | rows in the `predictions` table |

**A detail that explains confusing behaviour during a demo**: **Counters and
Histograms live in the process's memory** and reset to zero when the container
restarts. **Gauges read from SQLite** (`rakuten_predictions_stored_total`, the
drift ones) **survive**, because they are recomputed from the database on
every scrape. If after a restart you see "0 requests" but "97 predictions
stored", that is not a contradiction: they are two different mechanisms.

## 3.9 Grafana

**What it is.** The visualisation layer. It connects to data sources
(Prometheus, here) and draws panels. It also evaluates alert rules.

**What it does here.** Two dashboards and two alerts, all **provisioned from
files**: they exist the moment the stack starts, nobody draws them by hand.
Part 6 walks them panel by panel.

**An important nuance after the change in §2.6**: the alerts **inform and do
not act**. They are evaluated, they are visible in the UI, and they call
nothing.

## 3.10 Airflow

**What it is in general.** A workflow orchestrator. You describe tasks and
their dependencies as a directed acyclic graph (**DAG**), and Airflow runs
them in the right order, at the right time, retrying what fails and recording
everything.

What it adds over a `cron` job:

- **Explicit dependencies** — "this only if that finished successfully".
- **Retries with a policy** — count and spacing configurable.
- **Visibility** — a UI showing what ran, when, how long it took and why it
  failed, with per-task logs.
- **Backfill** — running the past if needed.
- **Cross-DAG triggering** — one flow can launch another.

**What it does here.** Two DAGs, and they deliberately do very little: every
task is an HTTP call (§2.5).

### `rakuten_training` — Sundays at 03:00, and whenever drift asks

```
start_training ──▶ wait_for_training ──▶ report_registry_decision ──▶ reload_champion
POST /training/    GET /training/status   GET /training/status         POST /model/reload
202, no wait       sensor, reschedule     reads the registry's         the API serves
                                          verdict                      the new champion
```

### `rakuten_drift_check` — hourly

```
run_drift_check ──▶ drift_above_threshold ──▶ trigger_training
POST /run           ShortCircuitOperator       TriggerDagRunOperator
on the drift        stops here unless          wait_for_completion=False
service             0.5 is crossed
```

The three operators involved, explained:

- **`PythonOperator`** — runs a Python function. Here, always one that makes
  an HTTP request.
- **`PythonSensor`** — waits for a condition, checking it periodically. In
  `mode="reschedule"` it frees the slot between checks instead of holding it.
- **`ShortCircuitOperator`** — if its function returns false, everything
  downstream is skipped. It is the "if there is no drift, do nothing".
- **`TriggerDagRunOperator`** — launches another DAG.

**Where to see it.** <http://localhost:8080> (admin / admin). Remember they are
born paused.

## 3.11 Streamlit

**What it is.** A library that turns a Python script into a web application,
with no HTML or JavaScript.

**What it does here.** The demonstration interface, in four sections:

1. **Overview & architecture** — the diagram, how a model reaches production,
   and the table of four faults found in the starting repository.
2. **Live prediction** — classify rows and see the result with its
   `model_version`.
3. **Model & registry** — version served, prediction count, and a button to
   start a training run.
4. **Monitoring** — the latest drift report and the operational metrics.

Important: **Streamlit does not load the model**. It calls the API over HTTP,
like any other client. That is consistent with the principle in §2.5.

## 3.12 GitHub Actions

**What it is.** GitHub's continuous integration: every push automatically runs
whatever you tell it to.

**What it does here.** Two jobs:

- **`lint-and-test`** — the linter and the full test suite.
- **`build-images`** — builds the four own images (`api`, `streamlit`,
  `drift`, `airflow`) in parallel, to check the Dockerfiles are still valid.

**What the tests protect.** 56 in total. Beyond the obvious, they pin
decisions that would otherwise be undone silently:

- That the training and serving paths never diverge again (the four faults in
  the original repository).
- That no DAG imports TensorFlow, Evidently or the training code.
- That no Grafana contact point calls the training endpoint.
- That `POST /run` accepts an empty body.

That last one deserves a comment, because it illustrates the philosophy:
`DriftRequest` declares all three fields with defaults, but FastAPI makes the
body required when the parameter is a Pydantic model without its own default.
The result was a 422 in response to the most obvious command in the world. The
DAG never hit it, because it always sends all three fields — so the only place
it was ever going to appear was a terminal, live, on the day of the defense.

---

# Part 4 — How everything flows

Two complete walkthroughs, following a concrete piece of data.

## 4.1 The journey of a prediction

1. Someone sends `POST /predict/` with a CSV path, an images path and a row
   limit.
2. The API checks a model is loaded. If not, 503.
3. It validates the limit against `RAKUTEN_MAX_PREDICT_ROWS` (200). Over it,
   422.
4. It reads the CSV **from inside the container**. This point is critical and
   explains a real failure: the path must be in a mounted directory (`data/`).
   A file in the host's system temp folder is invisible to the container, and
   produces a 404.
5. For each row: it assembles the text, tokenises it, loads the image,
   preprocesses it, runs both branches, and combines them with the blend
   weights.
6. It increments `rakuten_predictions_total` per class and observes the
   confidence.
7. It **stores every prediction in SQLite**, with its computed text features.
   Wrapped in `try/except`: a failure here does not break the response.
8. It returns the predictions along with `model_version` and `model_source`.

Meanwhile, a middleware has counted the request and measured its latency.
Prometheus will collect both on its next scrape.

## 4.2 The full cycle: from drift to a new model

```
 1. Real traffic arrives at /predict/
         │
         ▼
 2. Every prediction is stored in the `predictions` table
         │
         ▼
 3. Hourly, rakuten_drift_check sends POST /run to the drift service
         │
         ▼
 4. Evidently compares:
      reference = random sample of `products` (training data)
      current   = the N most recent predictions
    over text_length, word_count and prdtypecode
         │
         ▼
 5. It writes an HTML report, logs the metrics to MLflow,
    and stores the summary in SQLite
         │
         ▼
 6. ShortCircuitOperator: is the drifted share >= 0.5?
         │
    ┌────┴────┐
   no         yes
    │          │
  stop    7. TriggerDagRunOperator launches rakuten_training
               │
               ▼
          8. POST /training/ → immediate 202
               │
               ▼
          9. The sensor waits (in reschedule mode, without blocking)
               │
               ▼
         10. training.py trains, logs the run to MLflow with the
             DVC hashes, and registers a new version
               │
               ▼
         11. promote_if_better: does it beat the champion on weighted F1?
               │
          ┌────┴────┐
         no         yes
          │          │
    challenger   champion
    production       │
    untouched        ▼
                12. POST /model/reload: the API downloads the new
                    artifacts and serves them
                         │
                         ▼
                13. rakuten_model_version changes in Grafana
```

**The sentence that sums up Part 4**: no step in this cycle is a person.

---

# Part 5 — The demo

## 5.1 Before you start

With time to spare, not on the fly:

- [ ] `docker compose up -d` — and wait for all seven to be `healthy`. Check
      with `docker compose ps`. The API takes **around eight minutes** to open
      its port: it loads both Keras branches before it listens. Until then the
      check is refused instantly ("Could not connect", `after 0 ms`), which
      looks like a crash and is not one. Do not measure too early, and bring
      the stack up well before the defence.
- [ ] Run this from a `cmd` or PowerShell window of its own, **not** from an
      editor's integrated terminal. Compose draws its progress with cursor
      control codes, and some terminals never repaint the final line: the
      command has finished and looks hung. Where there is no choice,
      `set COMPOSE_PROGRESS=plain` turns that drawing off.
- [ ] Know that `docker compose up -d` **does not return immediately**.
      Services with `condition: service_healthy` make Compose wait for
      `mlflow`'s check before starting its dependents. It is not a hang and
      must not be interrupted: interrupting leaves half-created containers.
- [ ] Open all seven addresses in the browser, one by one. If any of them
      answers nothing while the service is `healthy`, try `127.0.0.1` in
      place of `localhost` (see the note at the end of Part 6) and, if that
      works, use that form throughout the demo. Finding this out live costs
      a minute you do not have.
- [ ] Unpause both DAGs in Airflow (<http://localhost:8080>, admin / admin)
      and trigger each once by hand, so the grid view has history to show.
- [ ] Verify the API is serving from the registry, not from disk:
      `curl.exe http://localhost:8000/health` must say
      `"model_source":"mlflow-registry"`.
- [ ] Send some normal traffic, so the dashboards are not empty.
- [ ] Have the tabs open: Streamlit, MLflow, Airflow, both Grafana dashboards.
- [ ] Confirm the data is pushed to the remote: `dvc status -c` from the DVC
      environment must report nothing missing. If files remain, `dvc push`
      resumes where it left off.
- [ ] Check the latest MLflow run carries the `data.*` tags. If it does not,
      it was trained before DVC was configured: launch a small training run
      through the API and it will.
- [ ] Rehearse the whole thing twice, timed.

## 5.2 The script, with commands

All commands from the repository root, in `cmd`.

### Step 1 — A live prediction (Streamlit)

<http://localhost:8501> → **Live prediction** → classify 10 rows.

Point at the `model_version` in the response: *the API serves what the
registry blessed, not a file someone left on disk*.

It takes about 20 seconds on CPU. Say so while it runs, not afterwards.

### Step 2 — The model registry (MLflow)

<http://localhost:5000> → the `rakuten-classification` experiment.

Show several runs compared, the tags holding the DVC hashes, and the
`rakuten-fusion` model with its versions and the `champion` tag.

The point to land: a version is promoted **only if it wins**; if it loses it
stays a `challenger` and production is untouched.

### Step 3 — Normal traffic

```cmd
python scripts\simulate_traffic.py --mode normal --batches 6 --batch-size 20
```

120 predictions across six batches 30 seconds apart, so the Grafana panels
draw a **timeline** instead of an instantaneous spike. Around 5-8 minutes on a
two-core machine.

While it runs, show the **API health** dashboard filling up.

### Step 4 — Shifted traffic

```cmd
python scripts\simulate_traffic.py --mode shifted --batches 6 --batch-size 20
```

`shifted` mode restricts traffic to a handful of categories and truncates the
text to three words, emptying the description. It moves both monitored column
families at once: `prdtypecode` (the output mix narrows) and `text_length` /
`word_count` (the text is deliberately short).

Where that data comes from, in case anyone asks: `training.py` only ever
samples a few hundred rows per class out of the ~84,000 in `products`.
Everything else has never been touched by any training run — it is a held-out
pool by construction, with no extra preparation.

### Step 5 — The drift check, now

```cmd
curl.exe -X POST http://localhost:8100/run -H "Content-Type: application/json" -d "{\"current_limit\":120,\"reference_limit\":300,\"log_to_mlflow\":false}"
```

**Why those parameters and not a bare call.** The POST does six synchronous
things: it samples the reference with `ORDER BY RANDOM()` (which forces a scan
and sort of all ~84,000 rows), computes text features row by row, runs
Evidently's tests, writes a multi-MB HTML file, **uploads it to MLflow**, and
stores the summary. `log_to_mlflow: false` skips the upload, which is the
slowest step; `reference_limit: 300` trims the sampling. Live, the difference
can be between ten seconds and two minutes of awkward silence.

`current_limit: 120` scopes the window to the 120 predictions just sent. That
is pure shifted traffic, and the drift comes out clean.

**The move that demonstrates judgement.** Run it again with the default
window:

```cmd
curl.exe -X POST http://localhost:8100/run -H "Content-Type: application/json" -d "{\"current_limit\":500,\"log_to_mlflow\":false}"
```

At 500 rows the window includes the earlier normal traffic, the signal is
diluted, and it may not cross the threshold. That is **not a failure**: it is
window size mattering. In production, with thousands of predictions an hour,
500 rows is a few minutes; here it is several sessions — meaning "current"
would stop meaning current.

### Step 6 — Airflow closing the loop

<http://localhost:8080> → grid view.

Walk the three tasks of `rakuten_drift_check` and show that each is an HTTP
call. Explain why: if Airflow imported the training code, TensorFlow would
have to go into its image and the dependency conflict that forced the stack
apart would return.

To show a real training run, launch it with a small configuration from
"Trigger DAG w/ config":

```json
{"samples_per_class": 50, "epochs_lstm": 1, "epochs_vgg": 1}
```

### Step 7 — The dashboards

Both, following the walkthrough in Part 6.

## 5.3 What to say when something is slow

- **`/predict/` is slow** → "every row runs a VGG16 forward pass on CPU; it is
  slow by nature, and that is why the training endpoint returns 202 instead of
  holding the connection open".
- **Streamlit is sluggish during the simulation** → "the API container is
  using both cores on inference; that is CPU contention, not a fault".
- **A panel says "No data"** → Part 6.4.

---

# Part 6 — The Grafana dashboards, panel by panel

<http://localhost:3000> — admin / admin. Both refresh every 30 seconds and
show the last hour by default.

## 6.1 The «API health» dashboard

Traffic, latency, errors and service readiness.

| Panel | What it shows | How to read it |
|---|---|---|
| **Requests / s by route** | Requests per second, split by route | Lets you tell prediction traffic apart from status polling |
| **p95 latency by route** | The 95th percentile of latency, per route | 95 % of requests are faster than this. `/predict/` is slow by nature: watch it against itself, not against a web-app baseline |
| **Error rate** | Share of 4xx and 5xx responses | Green below 5 %, orange to 20 %, red above |
| **Model loaded** | READY or NOT LOADED | The API starts even without a model; this is the real readiness signal |
| **Training in progress** | TRAINING or idle | Set while a background training run executes |
| **Responses by status** | Requests per second, by status code | Where the errors are, when there are any |
| **Completed training runs** | Finished runs per hour, by outcome | Only counts runs launched **through the API**; see 6.4 |
| **Total requests served** | Cumulative since the container last started | Resets on restart: it is an in-memory counter |

## 6.2 The «Model & data drift» dashboard

The model's state, and the relationship between what it sees now and what it
was trained on.

| Panel | What it shows | How to read it |
|---|---|---|
| **Dataset drift** | DRIFT / NO DRIFT | Evidently's latest verdict |
| **Drifted columns** | Share of columns that drifted | The retraining threshold is 0.5. With three monitored columns the possible values are 0 %, 33.3 %, 66.7 % and 100 % — which is why "33.3 %" means exactly "one of three" |
| **Drift report age** | Age of the latest report | If it climbs without resetting, the monitor has stopped running |
| **Serving model version** | The registry version number | **0 means the registry was unreachable and the local directory is being served.** Not a loud error, a labelled degradation |
| **Predicted class distribution** | Cumulative predictions per class | A collapse onto very few classes is the first visible sign that something is wrong. The labels are Rakuten's numeric codes; the dataset ships no names. There are 27 possible, so the panel is bounded even though the query does not limit it |
| **Mean prediction confidence** | Average confidence | Near 1/27 ≈ 0.037 would mean the model is barely better than guessing across 27 classes |
| **Predictions / s** | Prediction rate | Rises during the traffic simulation |
| **Predictions stored for monitoring** | Rows in the `predictions` table | This is the "current" dataset the drift check uses. **It survives restarts**, because it is read from SQLite |

## 6.3 The two alerts

Both provisioned from file, visible under Alerting → Alert rules.

- **«Data drift above retraining threshold»** — fires when
  `rakuten_drift_share_of_drifted_columns > 0.5` holds for 5 minutes. The same
  threshold the DAG uses, read from the same metric: what a person watches
  fire here is exactly the condition the orchestrator acts on.
- **«Drift monitor has stopped reporting»** — fires if there is no report in
  24 hours. This matters as much as the first: **a drift monitor that stopped
  running looks exactly like a healthy system**. Its silence is itself the
  alert.

**Both inform; neither acts.** That is a decision, not an oversight (§2.6).

## 6.4 When a panel says «No data»

Two different cases, worth not confusing:

- **A series that has never existed.** If no HTTP error has ever occurred, the
  error series is empty — and dividing by an empty vector in PromQL gives "No
  data", not zero. The fix is to wrap the numerator with `or vector(0)`, which
  is what the *Error rate* panel now does.
- **A real metric that is still zero.** *Completed training runs* only
  increments when a run is launched **through `POST /training/`**. If every
  training run was done with `python src/training.py` from the command line,
  the counter is legitimately empty. Not a fault: a metric measuring exactly
  what it says it measures.

---

# Part 7 — Likely questions

**"Why not PostgreSQL?"**
For the data, because the brief asks for "SQL or NoSQL locally" and SQLite is
a full SQL database that fits a single-node deployment. For Airflow's
metadata, because that database is its internal bookkeeping and has no
relationship to the project's data. In both cases, migrating is configuration,
not redesign: the access layer is isolated in `db.py`.

**"A teammate had a model scoring 0.9151 F1. Why is it not in production?"**
Because they are judged on different axes: 4.5 GB of weights and several
seconds per prediction versus 150 MB and under a second. For a project
assessed on MLOps criteria, the light one is the defensible choice. The right
home for the stronger one is the Model Registry, as a competing version —
exactly the comparison a registry exists to make.

**"Why does Airflow do so little?"**
Because doing more would mean installing TensorFlow and Evidently in its
image, recreating the dependency conflict that forced the services apart.
Airflow decides **when** and in **what order**; the services decide **how**.
Tests fail if a DAG imports anything heavy.

**"You have drift alerts. Why don't they trigger the retrain?"**
They did, until Airflow existed. A Grafana webhook posted to `/training/`.
Once the orchestrator arrived, that became a second automatic trigger for the
same action, on a different schedule, active whether or not the DAGs were
unpaused. The `/training/` lock kept them from colliding, which hid the
ambiguity rather than resolving it: when a run started, nothing recorded which
had started it. Alerts inform, Airflow acts.

**"What happens if MLflow goes down?"**
The API still starts, serves from the local directory and declares it in
`/health` as `model_source: local-directory`. Training continues, untracked.
Both are deliberate: observability should degrade, not take production with
it. And the degraded path is **labelled**, which is what caught a real failure
during development.

**"How do you know the registry is really being used?"**
`/health` and `/model-info` report the version and its source, and the startup
log shows the artifacts being downloaded. The fallback path is visible
precisely because it is labelled differently.

**"What exactly does 'drift' mean here?"**
Three columns. `text_length` and `word_count` are the same quantity on both
sides: if they move, the products arriving no longer look like the training
ones. `prdtypecode` compares **true** labels in the reference against
**predicted** ones in the current window: if it moves, the output mix changed,
which can mean the traffic changed or the model degraded. It is a symptom, not
a diagnosis. And ground-truth drift is not measurable: live predictions carry
no label.

**"How does the team share the exact data version?"**
The `.dvc` files are versioned in the repository and the remote lives on
DagsHub, so whoever clones runs `dvc pull` and retrieves exactly the same
data: the three CSVs and the images. What travels in the repository is the
remote's URL (`.dvc/config`), not the credentials — those live in
`.dvc/config.local`, which is ignored. Each person uses their own token.

**"Why DagsHub and not Google Drive?"**
Because Google blocks DVC's default OAuth application. It is not a warning
you can click past: the app requests full access to the user's Drive and DVC
has not managed to pass Google's verification; its maintainer describes the
situation as "stuck in limbo". The alternative was setting up our own Google
Cloud project, with tokens expiring after 7 days in testing mode. DagsHub
uses token authentication, with no consent screen and no application a third
party can block, and gives 100 GB free.

**"How would you scale this?"**
The API is stateless apart from the model it loads, so it scales horizontally
behind a load balancer. The real bottleneck is the VGG16 forward pass on CPU:
before replication, a GPU node or a batching layer.

**"Why `SequentialExecutor` if Airflow warns it is not for production?"**
Because it is the right size for two DAGs with sporadic runs on a two-core
node already running seven containers. Moving to `LocalExecutor` and Postgres
is configuration, not redesign.

---

# Part 8 — Known limitations

Stated plainly, because an acknowledged limitation is worth more than a hidden
one.

1. **Ground-truth drift cannot be measured.** Live predictions carry no
   labels, so accuracy in production is unobservable. Closing that gap would
   need a feedback loop — someone correcting or confirming the
   classifications — which is out of scope.

2. **Airflow runs on `SequentialExecutor` and SQLite.** One task at a time, no
   worker fleet. Right for a single-node demo, wrong for anything parallel.

3. **The DAGs are born paused.** Deliberate — starting the stack must not
   launch a multi-hour run — but it means remembering to unpause them.

4. **Model artifacts are versioned in two places.** The MLflow Model Registry
   is the source of truth, but `models/` is also tracked in git (the
   `/models/` line in `.gitignore` is commented out), so every training run
   dirties the repository. A real tension, still unresolved — and the natural
   answer is DVC, which is already set up: model artifacts are exactly the
   kind of large binary file it exists for.

5. **DVC versions the data, not the pipeline.** There are pointers and
   hashes, but no `dvc.yaml` and no `dvc repro`. That is deliberate: a DVC
   pipeline would be a **second orchestrator** alongside Airflow, deciding
   when to retrain on different criteria. It is the same trap as the two
   automatic triggers (§2.6), and Airflow already has that job.

6. **The development machine is the bottleneck.** Two cores, seven containers.
   The two timeout failures found and fixed during development are a direct
   consequence, and a third could appear.

7. **Three tests skip in the local development environment.** One needs
   Airflow and two need Evidently, which live deliberately in their own
   images. Not a coverage gap: it is dependency isolation showing up in the
   test suite. CI does run them, in the containers where those dependencies
   exist.

---

## Quick URL reference

| What | Where | Credentials |
|---|---|---|
| API (interactive docs) | <http://localhost:8000/docs> | — |
| Streamlit | <http://localhost:8501> | — |
| MLflow | <http://localhost:5000> | — |
| Drift service | <http://localhost:8100/docs> | — |
| Airflow | <http://localhost:8080> | admin / admin |
| Prometheus | <http://localhost:9090> | — |
| Grafana | <http://localhost:3000> | admin / admin |

**If an address does not respond.** First check the service is `healthy`
(`docker compose ps`); if it is and the browser will not load, try
`127.0.0.1` in place of `localhost`.

The reason: the servers inside the containers listen on IPv4
(`--host 0.0.0.0`). On a machine where `localhost` resolves to the IPv6
address `::1` first — the default on Windows — the request can fail to
arrive even though the service is perfectly alive. Changing the address in
the URL is the whole fix; nothing in the stack needs touching.

It does not affect the containers' healthchecks, which use `127.0.0.1` from
the inside for exactly this reason.
