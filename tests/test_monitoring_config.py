"""Grafana observes. Airflow acts.

The drift alert used to retrain the model itself, through a Grafana webhook
pointed at the API's training endpoint. That was the right design while there
was no orchestrator. Once `rakuten_drift_check` ran hourly, it became a second
automatic trigger for the same action, on a different schedule, firing whether
or not the DAGs were unpaused — and because `/training/` serialises runs behind
a lock, the two never collided loudly enough to be noticed. Nothing in a
started run recorded which of them had started it.

The failure mode these tests guard is quiet: re-adding a webhook here does not
break anything, it just restores an invisible second trigger. Nothing else in
the stack would fail, so nothing else would tell anyone.

They read YAML rather than running Grafana, which is the point — the dev
environment has no Grafana, and these are statements about configuration, not
about a running server.
"""

import os

import pytest

yaml = pytest.importorskip("yaml")

ROOT = os.path.join(os.path.dirname(__file__), "..")
ALERTING = os.path.join(ROOT, "monitoring", "grafana", "provisioning", "alerting")

CONTACT_POINTS = os.path.join(ALERTING, "contact-points.yml")
RULES = os.path.join(ALERTING, "rules.yml")

# The retrain threshold the drift service and the DAG both test. The alert is
# only useful as a human-readable mirror of the automated decision if it reads
# the same metric at the same cut-off.
DRIFT_THRESHOLD = 0.5
DRIFT_METRIC = "rakuten_drift_share_of_drifted_columns"


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_receivers(config):
    for contact_point in config.get("contactPoints") or []:
        for receiver in contact_point.get("receivers") or []:
            yield receiver


def test_no_contact_point_calls_the_api():
    """An alert may not reach into the API and start work."""
    for receiver in _iter_receivers(_load(CONTACT_POINTS)):
        url = str((receiver.get("settings") or {}).get("url", ""))
        assert "/training/" not in url, (
            f"a contact point posts to {url}. Grafana alerts inform; Airflow "
            "retrains. Two automatic triggers for one action means no run can "
            "say what started it."
        )
        assert "api:8000" not in url, (
            f"a contact point posts to the API at {url}. The orchestrator "
            "calls the services; the dashboards do not."
        )


def test_the_old_webhook_is_explicitly_deleted():
    """Grafana's alerting provisioning is additive.

    Removing a contact point from this file leaves the one already written to
    the database alone, and that database lives in the `grafana-data` volume,
    which survives `docker compose down`. On any stack that ran the old config
    even once — which is every stack this project has ever run — the webhook
    keeps firing from a file that no longer mentions it. Only an explicit
    delete removes it.
    """
    config = _load(CONTACT_POINTS)

    deleted = {
        entry.get("uid") for entry in config.get("deleteContactPoints") or []
    }
    assert "rakuten-retrain" in deleted, (
        "contact-points.yml must explicitly delete the `rakuten-retrain` "
        "receiver. Dropping its definition is not enough: provisioning is "
        "additive, so an already-provisioned webhook survives in the volume."
    )

    assert 1 in (config.get("resetPolicies") or []), (
        "the notification policy that routed to the webhook must be reset too, "
        "for the same reason — deleting a receiver does not remove a route."
    )


def test_the_drift_alert_mirrors_the_threshold_the_dag_acts_on():
    """The alert is the human-readable view of the automated decision.

    If the two drift apart, a presenter reads one number off a dashboard while
    the orchestrator acts on another.
    """
    rules = [
        rule
        for group in _load(RULES)["groups"]
        for rule in group["rules"]
    ]

    drift_rule = next(
        (rule for rule in rules if rule["uid"] == "rakuten-drift-threshold"),
        None,
    )
    assert drift_rule, "the drift alert is gone"

    queries = [item["model"] for item in drift_rule["data"]]

    expressions = [q.get("expr") for q in queries if q.get("expr")]
    assert DRIFT_METRIC in expressions, (
        f"the drift alert should read {DRIFT_METRIC}, the same series the "
        f"drift service publishes and the DAG reads. Found: {expressions}"
    )

    thresholds = [
        param
        for query in queries
        for condition in query.get("conditions") or []
        for param in (condition.get("evaluator") or {}).get("params") or []
    ]
    assert DRIFT_THRESHOLD in thresholds, (
        f"the alert fires at {thresholds}, but the drift service retrains at "
        f"{DRIFT_THRESHOLD} (RAKUTEN_DRIFT_THRESHOLD). A dashboard that "
        "disagrees with the orchestrator is worse than no dashboard."
    )


def test_the_staleness_alert_survives():
    """A drift monitor that stopped running looks exactly like a healthy one.

    This alert is the only thing that distinguishes them, which makes it the
    easiest one to delete by accident and the worst one to lose.
    """
    uids = {
        rule["uid"]
        for group in _load(RULES)["groups"]
        for rule in group["rules"]
    }
    assert "rakuten-drift-stale" in uids, (
        "the staleness alert is gone; silence from the monitor is now "
        "indistinguishable from a healthy system"
    )
