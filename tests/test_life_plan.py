from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stats_projections import simulate
from stats_projections.config import ScenarioError, compose
from stats_projections.engine import event_years
from stats_projections.life_plan import build, checks, describe

ROOT = Path(__file__).resolve().parent.parent
DIRS = dict(profile=ROOT / "profiles" / "my_profile.yaml", career_dir=ROOT / "careers",
            location_dir=ROOT / "locations", strategy_dir=ROOT / "strategies")


def mk(items, **over):
    plan = {"name": "P", "career": "rd_engineer", "location": None, "strategy": "work_to_65",
            "items": items, **over}
    return replace(build(plan, **DIRS), paths=300)


def years(s):
    return {k: set(np.unique(v).tolist()) for k, v in event_years(s).items()}


def test_empty_plan_matches_compose():
    a = mk([])
    b = replace(compose(DIRS["profile"], ROOT / "careers" / "rd_engineer.yaml",
                        ROOT / "strategies" / "work_to_65.yaml", name="P"), paths=300)
    np.testing.assert_allclose(simulate(a).net_worth, simulate(b).net_worth)


def test_items_move_to_chosen_ages():
    s = mk([{"kind": "partner", "age": 27, "amount": 120000},
            {"kind": "child", "age": 30}, {"kind": "child", "age": 33},
            {"kind": "home", "age": 35, "amount": 800000}])
    a0 = s.start_age
    y = years(s)
    assert y["Partner moves in"] == {27 - a0}
    assert y["Child 1 born"] == {30 - a0} and y["Child 2 born"] == {33 - a0}
    assert y["Buy a home"] == {35 - a0}
    assert next(e for e in s.events if e.name == "Partner moves in").amount == 120000
    assert next(e for e in s.events if e.name == "Buy a home").amount == 800000
    assert {"Child 2: daycare", "Child 2: college"} <= {e.name for e in s.expenses}


def test_break_part_time_retire_and_money_events():
    base = simulate(mk([]))
    s = mk([{"kind": "break", "age": 35, "years": 2}, {"kind": "retire", "age": 60},
            {"kind": "windfall", "age": 40, "amount": 50000}])
    r = simulate(s)
    a0 = s.start_age
    pay = r.income
    assert (pay[:, 35 - a0] < base.income[:, 35 - a0]).all()
    assert np.allclose(r.hours[:, 35 - a0:37 - a0], 0)
    assert (r.hours[:, 37 - a0] > 0).all()                       # back after the break
    assert np.allclose(r.hours[:, 60 - a0:], 0)                  # retired at 60
    pt = simulate(mk([{"kind": "part_time", "age": 45, "amount": 50}]))
    ratio = pt.hours[:, 46 - a0] / base.hours[:, 46 - a0]
    np.testing.assert_allclose(ratio, 0.5)


def test_abroad_skips_location_and_bad_ages_fail():
    s = mk([{"kind": "abroad", "age": 30}], location="texas")
    assert not any("Texas" in e.name for e in s.events)
    with pytest.raises(ScenarioError):
        mk([{"kind": "child", "age": 10}])
    with pytest.raises(ScenarioError):
        mk([{"kind": "home", "age": 30}, {"kind": "home", "age": 40}])


def test_checks_and_describe():
    plan = {"name": "P", "career": "rd_engineer", "items": [
        {"kind": "partner", "age": 30}, {"kind": "child", "age": 32}]}
    cs = checks(plan)
    labels = [c[0] for c in cs]
    assert "Partner moves in: 3 years later" in labels
    assert "Child born: doesn't happen" in labels
    assert len(cs) == 6 and plan["items"][0]["age"] == 30        # original untouched
    assert describe(plan) == "partner moves in at 30; child born at 32"
