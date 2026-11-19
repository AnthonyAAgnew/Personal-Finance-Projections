import json

import pytest

from stats_projections.storage import (GistStore, LocalStore, StorageError, percentile_of,
                                       slug, to_todays_dollars, years_between)


def test_local_store_plans_snapshots_checkins(tmp_path):
    s = LocalStore(tmp_path / "saved")
    assert s.plans() == {} and s.snapshots() == [] and s.checkins() == []
    s.save_plan({"name": "Industry vs EP", "mode": "Careers", "saved": "2026-01-01"},
                {"date": "2026-01-01", "plan": "Industry vs EP", "options": []})
    s.save_plan({"name": "Texas", "saved": "2026-02-01"})
    assert list(s.plans()) == ["Texas", "Industry vs EP"]          # newest first
    assert len(s.snapshots()) == 1
    s.save_plan({"name": "Industry vs EP", "mode": "Strategies", "saved": "2026-03-01"})
    assert s.plans()["Industry vs EP"]["mode"] == "Strategies"      # same name overwrites
    s.delete_plan("Texas")
    assert list(s.plans()) == ["Industry vs EP"]
    s.add_checkin({"date": "2026-05-01", "net_worth": 70000})
    s.add_checkin({"date": "2026-03-01", "net_worth": 65000})
    s.add_checkin({"date": "2026-05-01", "net_worth": 71000})       # same day replaces
    assert [c["net_worth"] for c in s.checkins()] == [65000, 71000]
    s.delete_checkin("2026-03-01")
    assert len(s.checkins()) == 1


def test_slug():
    assert slug("Industry vs EP, Texas!") == "industry_vs_ep_texas"
    assert slug("***") == "plan"


def test_percentile_and_dollars():
    band = {"p10": [0, 10], "p25": [0, 25], "p50": [0, 50], "p75": [0, 75], "p90": [0, 90]}
    assert percentile_of(50, band, 1.0) == pytest.approx(50)
    assert percentile_of(37.5, band, 1.0) == pytest.approx(37.5)
    assert percentile_of(1000, band, 1.0) == 95.0
    assert percentile_of(-5, band, 1.0) == 5.0
    assert percentile_of(25, band, 0.5) == pytest.approx(50)        # halfway in time
    assert percentile_of(10, band, 3.0) is None                     # beyond the band
    assert to_todays_dollars(110.25, 2, 0.05) == pytest.approx(100)
    assert years_between("2026-01-01", "2027-01-01") == pytest.approx(1, abs=0.01)


class FakeGist(GistStore):
    """GistStore with the HTTP layer replaced by an in-memory gist."""

    def __init__(self):
        super().__init__("token", "abc")
        self.remote = {"README.md": "hi"}
        self.calls = []

    def _request(self, method, body=None):
        self.calls.append(method)
        if method == "PATCH":
            for name, f in body["files"].items():
                if f is None:
                    self.remote.pop(name)
                else:
                    self.remote[name] = f["content"]
        return {"files": {n: {"content": c} for n, c in self.remote.items()}}


def test_gist_store_round_trip_and_cache():
    g = FakeGist()
    g.save_plan({"name": "A", "saved": "2026-01-01"}, {"date": "2026-01-01", "plan": "A"})
    assert json.loads(g.remote["plan__a.json"])["name"] == "A"
    assert list(g.plans()) == ["A"] and len(g.snapshots()) == 1
    n = g.calls.count("GET")
    g.plans(); g.snapshots()
    assert g.calls.count("GET") == n                                # cached until a write
    g.delete_plan("A")
    g.delete_plan("missing")                                        # no error, no request
    assert "plan__a.json" not in g.remote


def test_gist_errors_are_storage_errors():
    g = GistStore("bad", "nope", timeout=0.01)
    g.API = "http://127.0.0.1:9/"                                   # nothing listens here
    with pytest.raises(StorageError):
        g.plans()
