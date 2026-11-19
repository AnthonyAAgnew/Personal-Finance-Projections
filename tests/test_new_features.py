from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stats_projections import simulate
from stats_projections.breakeven import levers, solve
from stats_projections.config import Equity, compose, from_dict
from stats_projections.edit import to_dict
from stats_projections.engine import equity_proceeds
from stats_projections.report import failure_modes, free_time, runway_metrics
from stats_projections.stress import presets

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profiles" / "my_profile.yaml"


def make(**over):
    raw = {
        "name": "t", "horizon_years": 20, "paths": 50, "seed": 1,
        "portfolio": {"starting_balance": 0,
                      "returns": {"mean": 0.0, "stdev": 0.0},
                      "inflation": {"mean": 0.0, "stdev": 0.0}},
        "income": [{"name": "Pay", "amount": 100_000}],
        "expenses": [{"name": "Spend", "amount": 40_000}],
    }
    for k, v in over.items():
        if k == "portfolio":
            raw["portfolio"] = {**raw["portfolio"], **v}
        else:
            raw[k] = v
    return from_dict(raw)


def test_cash_reserve_fills_first_then_invests():
    s = make(portfolio={"cash": {"reserve_months": 6, "mean": 0.0, "stdev": 0.0}})
    res = simulate(s)
    np.testing.assert_allclose(res.cash[:, 1], 20_000)        # 6 months of $40k spending
    np.testing.assert_allclose(res.portfolio[:, 1], 60_000)   # all savings still counted
    np.testing.assert_allclose(res.cash[:, 5], 20_000)        # topped up, not growing


def test_runway_window_holds_more_cash_then_releases_it():
    s = make(portfolio={"cash": {"reserve_months": 0, "mean": 0.0, "stdev": 0.0,
                                 "runway_months": 24, "runway_window": [3, 6]}})
    res = simulate(s)
    assert res.cash[0, 3] == 0
    np.testing.assert_allclose(res.cash[:, 4], 80_000)        # 2 years of spending
    assert res.cash[0, 7] == 0                                # swept back after the window
    assert runway_metrics(res)["p_cash_out"] == 0


def test_glide_path_lowers_volatility_late():
    base = compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml", paths=2000)
    stocks = replace(base, allocation=())
    r_mix = simulate(base).real_returns
    r_all = simulate(stocks).real_returns
    assert r_mix[:, 50].std() < 0.8 * r_all[:, 50].std()


def test_stress_crash_hits_the_right_year_only():
    s = make(portfolio={"starting_balance": 100_000})
    crash = replace(s, stress=((5, -0.4, 0.0),))
    a, b = simulate(s), simulate(crash)
    np.testing.assert_allclose(b.real_returns[:, 5], -0.4)
    np.testing.assert_allclose(b.real_returns[:, 6], a.real_returns[:, 6])
    assert set(presets(4, 43)) and all(isinstance(v, tuple) for v in presets(4, 43).values())


def test_true_coast_stops_saving_but_keeps_hours():
    prof = [PROFILE, ROOT / "careers" / "rd_engineer.yaml"]
    true = simulate(compose(*prof, ROOT / "strategies" / "coast_true.yaml", paths=500))
    half = simulate(compose(*prof, ROOT / "strategies" / "coast_fire.yaml", paths=500))
    hit = true.strategy_year >= 0
    assert hit.mean() > 0.5
    y = int(np.median(true.strategy_year[hit])) + 2
    assert np.median(true.hours[hit, y]) > 40                 # still full time
    assert np.median(half.hours[:, y]) < 30                   # half hours
    rows = np.flatnonzero(hit & (true.strategy_year + 1 < 40))
    after = true.cash_flow[rows, true.strategy_year[rows] + 1]
    assert np.all(after <= 1e-6)                              # nothing saved once coasting


def test_equity_cap_table_and_preference():
    eq = Equity(sale_median=100e6, sale_spread=0.0, founder_share=0.5, option_pool=0.0,
                dilution=(0.5,), raised=80e6, state_rate=0.0, federal_rate=0.0)
    s = make(events=[{"name": "Exit", "type": "windfall", "year": 3, "equity": {
        "sale_median": 100e6, "sale_spread": 0.0, "founder_share": 0.5, "option_pool": 0.0,
        "dilution": [0.5], "raised": 80e6, "state_rate": 0.0, "federal_rate": 0.0}}])
    e = s.events[0]
    # Investors own 50%: converting gives $50M < $80M preference, so they take $80M;
    # common (50%) splits $20M, and you hold half of common.
    np.testing.assert_allclose(equity_proceeds(s, e), 10e6)
    assert eq.stake == pytest.approx(0.25)
    assert from_dict(to_dict(s)).events == s.events


def test_low_confidence_widens_outcomes():
    s = make(income=[{"name": "Pay", "amount": 100_000, "confidence": "low"}])
    pay = simulate(replace(s, paths=4000)).income[:, 0]
    assert 0.2 < np.log(pay).std() < 0.3
    assert np.allclose(pay, pay[:, None].mean(axis=1))        # persistent within a path


def test_home_counts_value_and_mortgage():
    s = make(events=[{"name": "Home", "type": "home", "amount": 500_000, "year": 2,
                      "down": 0.2, "rate": 0.0, "term": 10, "owner_costs": 0.02}],
             portfolio={"starting_balance": 200_000})
    res = simulate(s)
    np.testing.assert_allclose(res.home[:, 3], 500_000)
    np.testing.assert_allclose(res.debt[:, 3], 400_000 - 40_000)  # first payment made
    # Net worth = liquid + home - mortgage; the down payment just moved into the house.
    np.testing.assert_allclose(res.net_worth[:, 3], res.portfolio[:, 3] + 500_000 - 360_000)


def test_failure_modes_and_free_time():
    res = simulate(compose(PROFILE, ROOT / "careers" / "cardiology_ep.yaml", paths=500))
    fm = failure_modes(res)
    assert fm["crunch"]["p"] == pytest.approx(res.p_fail)
    assert fm["retirement"]["p"] <= fm["crunch"]["p"]
    ft = free_time(res)
    assert 0 < ft.free_share < 1 and ft.free_before_65 < ft.lifetime_free


def test_breakeven_finds_pay_that_matches():
    a = make(income=[{"name": "Pay", "amount": 80_000}])
    b = make(income=[{"name": "Pay", "amount": 100_000}])
    lv = next(x for x in levers(a) if "Pay" in x.label)
    be = solve(a, b, lv, at_age=20, paths=50)
    assert be.value == pytest.approx(100_000, rel=0.01)
