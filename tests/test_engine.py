from pathlib import Path

import numpy as np
import pytest

from stats_projections import ScenarioError, compare, load_scenario, simulate, summarize
from stats_projections.config import from_dict
from stats_projections.report import wilson_interval

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


def make(**over):
    """A tiny deterministic scenario: no volatility unless asked for."""
    raw = {
        "name": "t",
        "horizon_years": 10,
        "paths": 200,
        "seed": 1,
        "portfolio": {
            "starting_balance": 1000,
            "returns": {"model": "normal", "mean": 0.0, "stdev": 0.0},
            "inflation": {"mean": 0.0, "stdev": 0.0},
        },
        "income": [{"name": "Pay", "amount": 100}],
        "expenses": [{"name": "Spend", "amount": 60}],
        "events": [],
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k] = {**raw[k], **v}
        else:
            raw[k] = v
    return from_dict(raw)


# --- deterministic arithmetic -------------------------------------------------


def test_zero_volatility_matches_hand_calculation():
    res = simulate(make())
    np.testing.assert_allclose(res.net_worth[0], 1000 + 40 * np.arange(11))
    assert not res.failed.any()


def test_real_return_compounds_and_deflates():
    s = make(
        portfolio={
            "starting_balance": 1000,
            "returns": {"model": "normal", "mean": 0.05, "stdev": 0.0},
            "inflation": {"mean": 0.02, "stdev": 0.0},
        },
        income=[], expenses=[],
    )
    res = simulate(s)
    real = 1.05 / 1.02
    np.testing.assert_allclose(res.net_worth[0, -1], 1000 * real**10)


def test_purchase_and_failure_year():
    s = make(events=[{"name": "Boat", "type": "purchase", "amount": 1200, "year": 2}])
    res = simulate(s)
    # Year 0: 1040, year 1: 1080, year 2: 1120 - 1200 = -80 -> fails in year 2.
    assert res.failed.all()
    assert (res.first_failure_year == 2).all()
    assert res.net_worth[0, 3] == pytest.approx(-80)


def test_set_amount_step_change_and_stream_end():
    s = make(
        income=[{"name": "Pay", "amount": 100, "end_year": 8}],
        events=[{"name": "Raise", "type": "set_amount", "target": "Pay", "amount": 200, "year": 4}],
    )
    flows = simulate(s).cash_flow[0]
    np.testing.assert_allclose(flows, [40] * 4 + [140] * 4 + [-60] * 2)


def test_stream_growth_starts_from_its_start_year():
    s = make(income=[{"name": "Pay", "amount": 100, "start_year": 3,
                      "growth": {"mean": 0.10}}], expenses=[])
    flows = simulate(s).cash_flow[0]
    np.testing.assert_allclose(flows[:3], 0)
    np.testing.assert_allclose(flows[3:6], [100, 110, 121])


def test_negative_balance_accrues_borrowing_rate():
    s = make(
        income=[], expenses=[], borrowing_rate=0.10,
        portfolio={"starting_balance": -100,
                   "returns": {"model": "normal", "mean": 0.5, "stdev": 0.0},
                   "inflation": {"mean": 0.0, "stdev": 0.0}},
    )
    assert simulate(s).net_worth[0, 2] == pytest.approx(-121)


def test_failure_floor():
    assert not simulate(make()).failed.any()
    assert simulate(make(failure_floor=1050)).failed.all()


# --- events with timing uncertainty ------------------------------------------


def test_jitter_stays_in_range_and_same_time_as_follows():
    s = make(
        horizon_years=20,
        events=[
            {"name": "Buy", "type": "purchase", "amount": 10, "year": 5, "jitter": 2},
            {"name": "After", "type": "set_amount", "target": "Spend", "amount": 0,
             "same_time_as": "Buy", "offset": 3},
        ],
    )
    from stats_projections.engine import event_years

    y = event_years(s)
    assert y["Buy"].min() >= 3 and y["Buy"].max() <= 7
    assert len(np.unique(y["Buy"])) == 5
    np.testing.assert_array_equal(y["After"], y["Buy"] + 3)


def test_event_after_horizon_is_ignored():
    s = make(events=[{"name": "Late", "type": "purchase", "amount": 1e9, "year": 50}])
    assert not simulate(s).failed.any()


# --- reproducibility and common random numbers --------------------------------


def volatile(**over):
    return make(
        paths=2000, horizon_years=30,
        portfolio={"starting_balance": 1000,
                   "returns": {"model": "normal", "mean": 0.06, "stdev": 0.18},
                   "inflation": {"mean": 0.02, "stdev": 0.01}},
        income=[{"name": "Pay", "amount": 100, "growth": {"mean": 0.01, "stdev": 0.05},
                 "loss_probability": 0.05}],
        expenses=[{"name": "Spend", "amount": 95}],
        **over,
    )


def test_same_seed_same_result_and_different_seed_differs():
    a, b = simulate(volatile()), simulate(volatile())
    np.testing.assert_array_equal(a.net_worth, b.net_worth)
    c = simulate(volatile(seed=2))
    assert not np.array_equal(a.net_worth, c.net_worth)


def test_common_random_numbers_shared_across_scenarios():
    base = simulate(volatile())
    other = simulate(volatile(events=[{"name": "Gift", "type": "windfall",
                                       "amount": 500, "year": 1}]))
    np.testing.assert_array_equal(base.real_returns, other.real_returns)
    # A pure windfall can only help: no path fails that did not fail before.
    assert not (other.failed & ~base.failed).any()
    d = compare(base, other)
    assert d.delta <= 0 and d.ci[0] <= d.delta <= d.ci[1]


def test_student_t_matches_requested_moments():
    s = make(paths=20000, horizon_years=20, income=[], expenses=[],
             portfolio={"starting_balance": 1,
                        "returns": {"model": "student_t", "mean": 0.07, "stdev": 0.15, "dof": 5},
                        "inflation": {"mean": 0.0, "stdev": 0.0}})
    from stats_projections.draws import market

    nom, _ = market(s.returns, s.inflation, s.seed, s.paths, s.horizon_years)
    assert nom.mean() == pytest.approx(0.07, abs=0.003)
    assert nom.std() == pytest.approx(0.15, rel=0.05)


def test_inflation_correlation():
    s = make(paths=20000, portfolio={"starting_balance": 1,
             "returns": {"mean": 0.07, "stdev": 0.15},
             "inflation": {"mean": 0.02, "stdev": 0.01, "correlation": -0.5}})
    from stats_projections.draws import market

    nom, inf = market(s.returns, s.inflation, s.seed, s.paths, s.horizon_years)
    assert np.corrcoef(nom.ravel(), inf.ravel())[0, 1] == pytest.approx(-0.5, abs=0.02)


def test_bootstrap_samples_only_history(tmp_path):
    csv = tmp_path / "hist.csv"
    csv.write_text("year,return,inflation\n2000,0.10,0.02\n2001,-0.20,0.03\n2002,0.05,0.01\n")
    s = make(portfolio={"starting_balance": 1,
             "returns": {"model": "bootstrap", "data": str(csv), "block_size": 2}})
    from stats_projections.draws import market

    nom, inf = market(s.returns, s.inflation, s.seed, s.paths, s.horizon_years)
    assert set(np.unique(nom)) <= {0.10, -0.20, 0.05}
    # Returns and inflation stay paired with their own year.
    pairs = set(zip(nom.ravel().tolist(), inf.ravel().tolist()))
    assert pairs <= {(0.10, 0.02), (-0.20, 0.03), (0.05, 0.01)}


# --- statistics ----------------------------------------------------------------


def test_wilson_interval_matches_design_doc_rule_of_thumb():
    lo, hi = wilson_interval(500, 10_000)  # 5% failure on 10,000 paths
    assert (hi - lo) / 2 == pytest.approx(0.0043, abs=0.0003)
    assert wilson_interval(0, 100)[0] == 0.0


def test_summary_bands_are_ordered():
    s = summarize(simulate(volatile()))
    assert (s.bands[10] <= s.bands[50]).all() and (s.bands[50] <= s.bands[90]).all()
    assert s.cumulative_failure[-1] == pytest.approx(s.p_fail)


# --- scenario files --------------------------------------------------------------


def test_example_scenarios_load_and_inherit():
    base = load_scenario(SCENARIOS / "baseline.yaml")
    early = load_scenario(SCENARIOS / "early_retire.yaml")
    home = load_scenario(SCENARIOS / "buy_home.yaml")
    assert early.name == "Retire at 55"
    assert {e.name: e.year for e in early.events}["Retire"] == 25
    assert early.stream("Healthcare in retirement").start_year == 25
    assert early.stream("Healthcare in retirement").amount == 8000  # inherited
    assert len(home.events) == len(base.events) + 3
    assert (base.seed, base.paths, base.horizon_years) == (home.seed, home.paths, home.horizon_years)


def test_remove_and_unknown_keys(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "income: [{name: Pay, amount: 1}, {name: Side, amount: 2}]\n"
    )
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nincome: [{name: Side, remove: true}]\n")
    assert [s.name for s in load_scenario(tmp_path / "b.yaml").income] == ["Pay"]

    (tmp_path / "bad.yaml").write_text("income: [{name: Pay, amont: 1}]\n")
    with pytest.raises(ScenarioError, match="unknown key"):
        load_scenario(tmp_path / "bad.yaml")


def test_circular_extends(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ScenarioError, match="circular"):
        load_scenario(tmp_path / "a.yaml")


def test_bad_target_rejected():
    with pytest.raises(ScenarioError, match="unknown target"):
        make(events=[{"name": "X", "type": "set_amount", "target": "Nope", "year": 1}])
