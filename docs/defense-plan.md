# Defence plan — 13 October

30 minutes: **15 presentation · 5 demo · 10 Q&A**. Every member must speak.

The presentation can run from the Streamlit app instead of slides — the app's
first page carries the architecture and the findings for exactly that reason.

---

## Running order (15 min, five speakers, 3 min each)

### 1 — Problem and objectives
*Streamlit: "Overview & architecture", top of the page.*

- Classify a product into one of 27 Rakuten category codes from its title,
  description and photograph. 84,916 training products, 13,812 for test.
- State the framing explicitly, because it shapes everything that follows:
  **the model is not the deliverable.** The deliverable is the machinery that
  takes a model from a training run to a monitored service and back.
- One sentence on the baseline: a multimodal LSTM + VGG16 ensemble inherited
  from the starter repository, kept deliberately small.

### 2 — Architecture and why it is split
*Streamlit: the architecture diagram.*

- Walk the diagram once, following the data: CSVs → SQLite → training →
  registry → API → predictions back into SQLite → drift service → Prometheus
  → Grafana → and Airflow closing the loop back to training.
- Seven services, and the reason they are separate is concrete rather than
  decorative: **the MLflow server and the trainer cannot share a Python
  environment.** SQLAlchemy and Alembic need `typing_extensions >= 4.6`;
  TensorFlow 2.13 pins `< 4.6`. Evidently brings its own web framework.
  Containerisation is what made the stack installable at all.
- This is the strongest single point in the presentation: a real constraint,
  discovered by hitting it, resolved by architecture.
- **Airflow is the same argument, one level up.** The two DAGs call services
  over HTTP and compute nothing themselves, so the Airflow image installs
  neither TensorFlow nor Evidently nor the MLflow server. An orchestrator
  that imported the training code would have re-created the very conflict
  that forced the split. `tests/test_dags.py` enforces the boundary, so it
  cannot erode quietly.

### 3 — Training, tracking and the registry
*MLflow UI: the experiment list, then the Models tab.*

- Every run logs parameters, metrics, the DVC hash of the data it used, and a
  self-contained artifact bundle.
- Promotion is a decision, not a side effect: the new version is scored on
  held-out weighted F1 against the reigning champion and is promoted only if
  it wins. Otherwise it stays a challenger and production does not move.
- Show the two versions side by side and say what the difference was: same
  hyperparameters, same data, different code. F1 went from 0.081 to 0.126.
- Mention the detail that makes the comparison honest: three disjoint
  splits, one job each. The networks fit on train, early stopping and the
  blend search both use validation because both are model selection, and the
  number that decides promotion is scored once on test.

### 4 — What we found in the baseline
*Streamlit: the mismatch table.*

This is the section that distinguishes the project. Four mismatches between
training and serving, none of which raised an error:

| Training wrote | Serving read | Effect |
|---|---|---|
| `best_weights.pkl` | `best_weights.json` | Retraining changed nothing at all |
| `mapper.pkl` | `mapper.json` | Stale class mapping |
| Raw 0–255 pixels | `vgg16.preprocess_input` | Image branch given weight 0.0 |
| title + description | description only | Title-only products scored as empty strings |

- Lead with the evidence for the fourth, because it is vivid: a puppet, a
  trading card, a pool pump and a paper shredder all returned the same class
  with confidence identical to sixteen decimal places. The model was
  receiving the same empty input for all four.
- Close on the fix that matters more than the bugs: each one is now pinned by
  a test that fails CI if the paths diverge again.
- If there is time, add the fifth finding, which came from a symptom rather
  than from reading code: the blend search kept giving the image branch a
  weight of exactly zero even though it scored better than the text branch.
  The weights were being tuned on training data — and underneath, validation
  turned out to be a subset of test, so nothing was genuinely held out.

### 5 — Monitoring and operations
*Grafana: both dashboards.*

- API health: traffic, status codes, p95 latency per route, readiness.
- Model and drift: predicted class distribution, mean confidence, served
  model version, drift verdict.
- Be precise about what drift means here: input drift and prediction drift
  are two different measurements, and **ground-truth drift is not measurable**
  because live predictions have no labels. Say so before anyone asks.
- Two alerts, and explain the second one: a drift monitor that stopped
  running looks exactly like a healthy system, so its silence is itself an
  alert.
- Both alerts inform; neither acts. Worth saying out loud, because it is a
  decision rather than an omission: the drift alert used to retrain through a
  Grafana webhook, from before Airflow existed. Keeping both would have meant
  two automatic triggers for one action, on different schedules, with nothing
  in a started run to say which one started it. One orchestrator, one path.

---

## Demo (5 min)

Rehearse this. Have the stack already running — `docker compose up -d` before
the session, not during it.

1. **Streamlit → Live prediction.** Score 10 rows. Point out `model_version`
   in the response: the API is serving what the registry blessed, not a file
   someone left on disk. (~20 s on CPU; say so while it runs.)
2. **Streamlit → Model & registry.** Show the serving version and the
   prediction count rising. Trigger a small training run; it returns
   immediately with 202 because a full run takes hours and an HTTP request
   should not be held open for that long.
3. **MLflow → Models.** Show the champion and challenger tags, and the metric
   that decided between them.
4. **Grafana.** The request you just made is already on the API health
   dashboard. Show the drift panel.
5. **Back to Streamlit → Monitoring.** The latest Evidently verdict.
6. **Airflow.** Show the two DAGs with green runs in the grid, then open
   `rakuten_drift_check` and walk the three tasks: check → gate → trigger.
   The sentence to land: *nothing in this loop is a person.* If time is
   short, this is the step to cut — but say the sentence anyway.

Fallback if anything is down: every screen has a static equivalent in the
README. Say what you intended to show and move on; do not debug live.

---

## Likely questions

**"The accuracy is very low."**
It is, and deliberately: one epoch on a subsample, trained on a CPU laptop.
The brief states the model is not what is assessed. What we can show is that
the pipeline detects an improvement and promotes it — v2 beat v1 by 56% on
weighted F1 and was promoted automatically, with no change to the serving
code. A better model would travel the same path without a single edit.

**"Why does Airflow do so little?"**
Because that is the design. Each task is one HTTP call to the service that
owns the work: `POST /training/`, poll `/training/status`, read the registry
verdict, `POST /model/reload`. Airflow decides *when* and *in what order*;
the services decide *how*. Keeping it that way is what lets the Airflow image
install none of the heavy dependencies — which matters here specifically,
because those dependencies are mutually incompatible. Two details worth
volunteering: the training sensor runs in `reschedule` mode, so a run that
lasts hours does not hold the executor's only slot, and the drift DAG fires
training without waiting, so the hourly schedule never backs up behind it.

**"How does drift actually cause a retrain?"**
`rakuten_drift_check` runs hourly, calls the drift service, and reads back
`action_required` — the threshold lives in the service, so there is one
definition in one place. A `ShortCircuitOperator` stops the DAG when it is
false; when it is true, `TriggerDagRunOperator` starts `rakuten_training`,
which retrains, lets the registry decide whether the new version is better,
and reloads the API only if it is. That is the full loop, and no step in it
is a person.

**"You have alerts on drift. Why don't they trigger the retrain?"**
They did, until Airflow existed. A Grafana webhook posted to `/training/`
when the drift alert fired — a reasonable design when there was no
orchestrator. Once `rakuten_drift_check` was running hourly, that webhook
became a second automatic trigger for the same action, on a different
schedule, firing whether or not the DAGs were unpaused. The `/training/`
lock kept them from overlapping, which hid the ambiguity rather than
removing it: when a run started, nothing recorded which of the two had
started it. The alert rules still exist and still use the same 0.5 threshold
the DAG tests, so a person sees exactly the condition the orchestrator acts
on. Alerts inform, Airflow acts.

**"Why SQLite?"**
It fits a single-node deployment and a database that is read far more than
written. The access layer is isolated in `db.py`, so moving to Postgres is a
connection change, not a rewrite.

**"What happens if MLflow is down?"**
The API falls back to the local model directory and reports
`model_source: local-directory` in `/health`. Training also continues
untracked rather than failing — a dead tracking server must not cost a
multi-hour run. Both are deliberate: observability should degrade, not take
production with it.

**"How do you know the registry is really being used?"**
`/health` and `/model-info` report the version and its source, and the
startup log shows the artifacts being downloaded. The fallback path is
visible precisely because it is labelled differently.

**"A teammate had a model scoring 0.9151 F1. Why is it not in production?"**
Because the two models are judged on different axes. That one is seven
fine-tuned encoders, roughly 4.5 GB of weights, several seconds per
prediction on CPU. Ours is around 150 MB and answers in well under a second.
For a project assessed on MLOps — image size, versioning cost, serving
latency, a demo that has to work live — the lighter model is the defensible
choice. The right home for the stronger one is the Model Registry as a
competing version, which is exactly the comparison the registry exists to
make.

**"How would you scale this?"**
The API is stateless apart from the model it loads, so it scales
horizontally behind a load balancer. The real bottleneck is the VGG16 forward
pass on CPU; a GPU node or a batching layer addresses that before any
replication does.

---

## Before the day

- [ ] Assign the five sections to names.
- [ ] `docker compose up -d --build` and confirm all seven services are healthy.
- [ ] Unpause both DAGs in Airflow and trigger each once, so the UI shows
      green runs rather than an empty grid.
- [ ] Serve a few predictions so the dashboards and drift job have data.
- [ ] Run the drift job once so the monitoring page is not empty.
- [ ] Rehearse the demo end to end at least twice, timed.
- [ ] Re-read the README for typos — it is explicitly graded.
