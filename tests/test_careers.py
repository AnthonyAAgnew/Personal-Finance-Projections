from pathlib import Path

import numpy as np
import pytest

from stats_projections import ScenarioError, simulate
from stats_projections.config import TaxLayer, compose, from_dict
from stats_projections.edit import to_dict
from stats_projections.engine import NEVER, event_years, loan_payment, progressive_tax
from stats_projections.report import career_metrics

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profiles" / "my_profile.yaml"
CAREERS = sorted((ROOT / "careers").glob("*.yaml"))


def make(**over):
    raw = {
        "name": "t", "horizon_years": 20, "paths": 4000, "seed": 7,
        "portfolio": {"starting_balance": 0,
                      "returns": {"mean": 0.0, "stdev": 0.0},
                      "inflation": {"mean": 0.0, "stdev": 0.0}},
        "income": [{"name": "Pay", "amount": 100}],
        "expenses": [],
        "events": [],
    }
    raw.update(over)
    return from_dict(raw)


# --- outcomes -------------------------------------------------------------------


def test_probability_controls_share_of_paths():
    s = make(events=[{"name": "Bonus", "type": "windfall", "amount": 1000, "year": 2,
                      "probability": 0.3}])
    got = (simulate(s).cash_flow[:, 2] > 500).mean()
    assert got == pytest.approx(0.3, abs=0.03)


def test_group_outcomes_are_mutually_exclusive_and_linked_events_follow():
    s = make(events=[
        {"name": "A", "type": "windfall", "amount": 1, "year": 3, "group": "g", "probability": 0.5},
        {"name": "B", "type": "windfall", "amount": 1, "year": 3, "group": "g", "probability": 0.3},
        {"name": "A2", "type": "windfall", "amount": 1, "same_time_as": "A", "offset": 2},
    ])
    y = event_years(s)
    a, b = y["A"] < NEVER, y["B"] < NEVER
    assert not (a & b).any()
    assert a.mean() == pytest.approx(0.5, abs=0.03) and b.mean() == pytest.approx(0.3, abs=0.03)
    np.testing.assert_array_equal(y["A2"] < NEVER, a)


def test_group_over_one_rejected():
    with pytest.raises(ScenarioError, match="sum above 1"):
        make(events=[
            {"name": "A", "type": "windfall", "year": 1, "group": "g", "probability": 0.7},
            {"name": "B", "type": "windfall", "year": 1, "group": "g", "probability": 0.7},
        ])


def test_spread_is_lognormal_with_median_amount():
    s = make(events=[{"name": "Exit", "type": "windfall", "amount": 1000, "year": 1,
                      "spread": 1.0, "tax_rate": 0.2}])
    got = simulate(s).cash_flow[:, 1] - 100
    assert np.median(got) == pytest.approx(800, rel=0.05)
    assert got.std() > 400


def test_duration_reverts_stream():
    s = make(events=[{"name": "School", "type": "set_amount", "target": "Pay", "amount": 0,
                      "year": 2, "duration": 3}])
    np.testing.assert_allclose(simulate(s).cash_flow[0, :7], [100, 100, 0, 0, 0, 100, 100])


def test_noise_keeps_mean():
    s = make(income=[{"name": "Pay", "amount": 100, "noise": 0.3}])
    flows = simulate(s).cash_flow
    assert flows.mean() == pytest.approx(100, rel=0.01)
    assert flows.std() > 20


# --- loans and taxes -----------------------------------------------------------------


def test_loan_cash_in_then_amortized_payments():
    s = make(income=[], events=[{"name": "L", "type": "loan", "amount": 1000, "year": 1,
                                  "rate": 0.05, "defer": 2, "term": 4}])
    flow = simulate(s).cash_flow[0]
    pay = loan_payment(np.array([1000.0]), 0.05, 2, 4)[0]
    np.testing.assert_allclose(flow[:8], [0, 1000, 0, -pay, -pay, -pay, -pay, 0])
    # Present value of the payments at the disbursement year equals the loan.
    pv = sum(pay / 1.05 ** (2 + k) for k in range(4))
    assert pv == pytest.approx(1000)


def test_progressive_tax_by_hand():
    layer = TaxLayer("t", ((0, 0.1), (100, 0.2), (200, 0.3)), deduction=50)
    got = progressive_tax(np.array([0.0, 100.0, 300.0]), (layer,))
    # 300 - 50 = 250 -> 10 + 20 + 15
    np.testing.assert_allclose(got, [0, 5, 45])


def test_taxable_income_needs_taxes():
    with pytest.raises(ScenarioError, match="no 'taxes'"):
        make(income=[{"name": "Pay", "amount": 100, "taxable": True}])


def test_tax_reduces_take_home_and_is_reported():
    s = make(income=[{"name": "Pay", "amount": 100000, "taxable": True}],
             taxes=[{"name": "flat", "brackets": [[0, 0.25]]}])
    res = simulate(s)
    assert res.cash_flow[0, 0] == pytest.approx(75000)
    assert res.tax[0, 0] == pytest.approx(25000) and res.income[0, 0] == 100000


# --- profile + careers -------------------------------------------------------------------


def test_every_career_composes_and_round_trips():
    for f in CAREERS:
        s = compose(PROFILE, f)
        assert s.stream("Career pay").taxable
        assert from_dict(to_dict(s)) == s, f.name


def test_careers_share_market_draws_and_differ_in_pay():
    a = simulate(compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml", paths=2000))
    b = simulate(compose(PROFILE, ROOT / "careers" / "regulator.yaml", paths=2000))
    np.testing.assert_array_equal(a.real_returns, b.real_returns)
    assert np.median(a.income[:, 10]) > np.median(b.income[:, 10])


def test_doctor_paths_split_by_admission():
    res = simulate(compose(PROFILE, ROOT / "careers" / "doctor.yaml", paths=4000))
    in_school = res.income[:, 8] < 1000  # year 8 is the second year of med school
    assert in_school.mean() == pytest.approx(0.6, abs=0.03)


def test_career_metrics():
    res = simulate(compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml", paths=2000))
    m = career_metrics(res, 65, 1_000_000)
    assert m.net_worth_at_age[10] <= m.net_worth_at_age[50] <= m.net_worth_at_age[90]
    assert 0 <= m.p_reach <= 1 and m.lifetime_income > m.lifetime_tax > 0
