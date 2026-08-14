"""
Tests for the two admin controls: rescore and embedding re-check.

    python3.11 -m pytest test_rescore.py -q

WHAT THESE GUARD. Rescore is sold on one property: it redoes the maths on text already
on disk and touches no network. That claim is what makes it safe to press after a
scoring change, and it is the claim a future edit is most likely to break -- one call to
a resolver slipped into record_snapshot and the button quietly becomes a full re-scrape.
So the network is not mocked to return something harmless here; it is mocked to RAISE.
"""

import unittest.mock as mk

import pytest

import main


class NetworkUsed(AssertionError):
    """Raised by every network entry point, so any call fails the test loudly."""


@pytest.fixture
def offline(monkeypatch):
    """Makes every outbound call an error rather than a stub."""
    def boom(*args, **kwargs):
        raise NetworkUsed("a network call was made")

    import requests
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "request", boom)
    return boom


def store(*names):
    master = {
        n: {
            "skill_name": n,
            "status": "approved",
            "category": "software",
            "wikipedia_summary": f"{n} is a piece of software used for doing things.",
            "onet_codes": [], "onet_titles": [], "occupations": [],
        }
        for n in names
    }
    return master, []


def run_rescore(monkeypatch, master, timeseries):
    """Runs the rescore against an in-memory store, capturing what it would save."""
    saved = {}
    monkeypatch.setattr(main, "load_master", lambda: master)
    monkeypatch.setattr(main, "load_timeseries", lambda: timeseries)
    monkeypatch.setattr(main, "save_master", lambda m: saved.update(master=m))
    monkeypatch.setattr(main, "save_timeseries", lambda t: saved.update(timeseries=t))
    main._run_rescore()
    return saved


# --- The property the button is sold on ---------------------------------------

def test_rescore_makes_no_network_calls(monkeypatch, offline):
    """
    The whole point. If this fails, pressing the button costs quota and re-fetches pages
    that have not changed, and the honest description of it on the review page becomes a
    lie.
    """
    master, timeseries = store("Alpha", "Beta")
    saved = run_rescore(monkeypatch, master, timeseries)
    assert len(saved["timeseries"]) == 2


def test_rescore_writes_a_snapshot_for_every_approved_skill(monkeypatch, offline):
    master, timeseries = store("Alpha", "Beta", "Gamma")
    saved = run_rescore(monkeypatch, master, timeseries)
    assert {r["skill_name"] for r in saved["timeseries"]} == {"Alpha", "Beta", "Gamma"}


def test_rescore_skips_anything_not_approved(monkeypatch, offline):
    """An unapproved skill is not on the dashboard, so it has no score to refresh."""
    master, timeseries = store("Alpha", "Pending")
    master["Pending"]["status"] = "pending"
    saved = run_rescore(monkeypatch, master, timeseries)
    assert {r["skill_name"] for r in saved["timeseries"]} == {"Alpha"}


def test_rescore_is_idempotent(monkeypatch, offline):
    """
    Twice in a row must be byte-identical. A button someone will press when unsure has to
    be safe to press twice, and one snapshot per skill per quarter is the invariant
    upsert_snapshot exists to hold.
    """
    master, timeseries = store("Alpha", "Beta")
    first = run_rescore(monkeypatch, master, timeseries)["timeseries"]
    snapshot = [dict(r) for r in first]

    second = run_rescore(monkeypatch, master, timeseries)["timeseries"]
    assert len(second) == 2, "a second run must not append duplicate snapshots"
    assert [dict(r) for r in second] == snapshot


def test_rescore_records_the_new_fields(monkeypatch, offline):
    """The fields the scoring change added have to survive the trip through the store."""
    master, timeseries = store("Alpha")
    saved = run_rescore(monkeypatch, master, timeseries)
    record = saved["timeseries"][0]
    for field in ("ai_score_base", "embedded_ai_boost", "lexical_ai_boost",
                  "lexical_ai_terms", "contrast_sim", "embeds_ai"):
        assert field in record, field


def test_rescore_records_a_deterministic_phrase_without_scoring_it(monkeypatch, offline):
    """
    End to end: the text says "machine learning", so the phrase is recorded on the
    snapshot. It adds nothing -- the generative pole already measures that phrase, and
    a boost on top would double-count it.
    """
    master, timeseries = store("Alpha")
    master["Alpha"]["wikipedia_summary"] = "Alpha is a machine learning toolkit."
    saved = run_rescore(monkeypatch, master, timeseries)
    record = saved["timeseries"][0]
    assert record["lexical_ai_terms"] == ["machine learning"]
    assert record["lexical_ai_boost"] == 0.0
    assert record["ai_score"] == record["ai_score_base"]


def test_a_busy_store_aborts_rather_than_queueing(monkeypatch, offline):
    """A job that silently waits is a job nobody knows is waiting."""
    import run_state
    master, timeseries = store("Alpha")
    monkeypatch.setattr(main, "load_master", lambda: master)
    monkeypatch.setattr(main, "load_timeseries", lambda: timeseries)

    def busy(*args, **kwargs):
        raise run_state.StoreBusy("something else is writing")

    monkeypatch.setattr(run_state, "store_writer", busy)
    main._run_rescore()          # must not raise
    assert timeseries == [], "nothing may be written while another writer holds the store"


# --- The re-check button ------------------------------------------------------

def test_recheck_refuses_an_empty_name():
    """It takes one name on purpose; a blank must not become "re-probe everything"."""
    response = main.handle_recheck_embedding(mk.Mock(), skill_name="   ")
    assert "recheck_err" in response.headers["location"]
    assert response.status_code == 303


def test_recheck_queues_the_named_skill():
    tasks = mk.Mock()
    response = main.handle_recheck_embedding(tasks, skill_name="Adobe Acrobat")
    tasks.add_task.assert_called_once()
    assert tasks.add_task.call_args[0][1] == "Adobe Acrobat"
    assert "Adobe%20Acrobat" in response.headers["location"]


def test_a_blocked_search_leaves_the_skill_unestablished(monkeypatch):
    """
    The three-state rule, enforced on this path too. A rate-limited search must not be
    recorded as "this product has no AI features" just because a person pressed a button.
    """
    import embedding_probe, storage
    master, timeseries = store("Alpha")
    master["Alpha"]["embeds_ai"] = None
    cache = {}

    monkeypatch.setattr(main, "load_master", lambda: master)
    monkeypatch.setattr(main, "load_timeseries", lambda: timeseries)
    monkeypatch.setattr(main, "save_master", lambda m: None)
    monkeypatch.setattr(main, "save_timeseries", lambda t: None)
    monkeypatch.setattr(storage, "load_embedding_cache", lambda: cache)
    monkeypatch.setattr(storage, "save_embedding_cache", lambda c: None)
    monkeypatch.setattr(
        embedding_probe, "search_embedding_evidence",
        lambda name: {"query": "", "results": [], "error": embedding_probe.ERROR_BLOCKED},
    )
    # Give it a snapshot so it is in scope.
    timeseries.append({"skill_name": "Alpha", "quarter": "2026Q3", "ai_score": 0.1,
                       "tech_base_sim": 0.0, "ml_pipeline_sim": 0.0, "embedded_ai_sim": 0.0})

    main._run_recheck_embedding("Alpha")

    assert cache["Alpha"]["embeds_ai"] is None
    assert cache["Alpha"]["error"] == embedding_probe.ERROR_BLOCKED
    assert master["Alpha"].get("embeds_ai") is None


def test_recheck_ignores_a_skill_that_is_not_in_the_store(monkeypatch):
    import storage
    monkeypatch.setattr(main, "load_master", lambda: {})
    monkeypatch.setattr(main, "load_timeseries", lambda: [])
    monkeypatch.setattr(storage, "load_embedding_cache", lambda: {})
    main._run_recheck_embedding("Nonexistent")   # must not raise
