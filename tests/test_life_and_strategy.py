from pathlib import Path

import numpy as np
import pytest

from stats_projections import simulate
from stats_projections.config import compose, from_dict
from stats_projections.edit import to_dict
from stats_projections.report import debt_metrics, hours_metrics, strategy_metrics

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profiles" / "my_profile.yaml"
CAREERS = sorted((ROOT / "careers").glob("*.yaml"))
ADDONS = sorted((ROOT / "life").glob("*.yaml")) + sorted((ROOT / "strategies").glob("*.yaml"))


def make(**over):
    raw = {
        "name": "t", "horizon_years": 20, "paths": 20, "seed": 1,
        "portfolio": {"starting_balance": 0,
                      "returns": {"mean": 0.0, "stdev": 0.0},
                      "inflation": {"mean": 0.0, "stdev": 0.0}},
        "income": [{"name": "Pay", "amount": 100_000}],
        "expenses": [{"name": "Spend", "amount": 40_000}],
    }
    raw.update(over)
    return from_dict(raw)


# --- debt counted in net worth ------------------------------------------------------


def test_loan_balance_reduces_net_worth_until_repaid():
    s = make(income=[], expenses=[], events=[
        {"name": "L", "type": "loan", "amount": 1000, "year": 1, "rate": 0.10, "defer": 2, "term": 3}])
    res = simulate(s)
    debt = res.debt[0]
    # Owed at the end of year 1: 1000; grows 10% through the deferral.
    np.testing.assert_allclose(debt[2], 1000)
    np.testing.assert_allclose(debt[3], 1100)
    assert debt[6] == pytest.approx(0, abs=1e-6)  # paid off after the third payment
    # The cash was received, so the portfolio is up but net worth is not.
    np.testing.assert_allclose(res.net_worth[0, 2], res.portfolio[0, 2] - 1000)


def test_med_school_debt_vs_no_debt():
    with_debt = simulate(compose(PROFILE, ROOT / "careers" / "doctor.yaml", paths=2000))
    no_debt = simulate(compose(PROFILE, ROOT / "careers" / "doctor_no_debt.yaml", paths=2000))
    dm = debt_metrics(with_debt)
    assert dm.p_borrow == pytest.approx(0.6, abs=0.03)  # only admitted paths borrow
    assert 300_000 < dm.peak_median < 450_000  # 4 x $75.7k grown at ~6% real
    assert debt_metrics(no_debt) is None
    # Same random draws: no debt can only be better at 45.
    t = 45 - 29
    assert np.median(no_debt.net_worth[:, t]) > np.median(with_debt.net_worth[:, t])


# --- taxes ---------------------------------------------------------------------------


def test_per_earner_payroll_cap():
    taxes = [{"name": "SS", "per_earner": True, "brackets": [[0, 0.1], [100_000, 0.0]]}]
    one = simulate(make(income=[{"name": "A", "amount": 200_000, "taxable": True}], taxes=taxes))
    two = simulate(make(income=[{"name": "A", "amount": 100_000, "taxable": True},
                                {"name": "B", "amount": 100_000, "taxable": True}], taxes=taxes))
    assert one.tax[0, 0] == pytest.approx(10_000) and two.tax[0, 0] == pytest.approx(20_000)


# --- scale events, hours, cost of living --------------------------------------------------


def test_scale_event_multiplies_after_set_amount():
    s = make(events=[{"name": "Remote", "type": "scale", "target": "Pay", "factor": 0.5, "year": 2},
                     {"name": "Raise", "type": "set_amount", "target": "Pay", "amount": 200_000,
                      "year": 4}])
    flows = simulate(s).cash_flow[0]
    np.testing.assert_allclose(flows[:6], np.array([100, 100, 50, 50, 100, 100]) * 1000 - 40_000)


def test_hours_follow_events_and_retirement():
    s = make(time=[{"name": "Work hours", "amount": 50}],
             events=[{"name": "Retire", "type": "set_amount", "target": ["Pay", "Work hours"],
                      "amount": 0, "year": 10}])
    res = simulate(s)
    assert res.hours[0, 9] == 50 and res.hours[0, 10] == 0
    hm = hours_metrics(res)
    assert hm.lifetime_hours == 50 * 48 * 10
    assert hm.take_home_per_hour == pytest.approx(100_000 * 10 / (50 * 48 * 10))


def test_cost_index_scales_tier_budgets():
    lifestyle = {"creep": 0.0, "withdrawal_rate": 0.0, "smoothing_years": 1,
                 "tiers": [{"name": "T", "take_home": 50_000, "monthly": {"Spend": 1000}}]}
    base = simulate(make(lifestyle=lifestyle))
    cheap = simulate(make(lifestyle={**lifestyle, "cost_index": [[3, 0.5]]}))
    assert base.spending[0, 5] == 12_000 and cheap.spending[0, 5] == 6_000
    assert cheap.spending[0, 2] == 12_000


# --- strategies --------------------------------------------------------------------------


def test_fire_stops_stop_streams_at_25x_spending():
    s = make(time=[{"name": "Work hours", "amount": 50}],
             strategy={"kind": "fire", "multiple": 5, "stop": ["Pay", "Work hours"],
                       "target_year": 19})
    res = simulate(s)
    # Saves 60k a year; 5 x 40k = 200k is reached at the end of year 4 (240k).
    assert res.strategy_year[0] == 4
    assert res.income[0, 3] == 100_000 and res.income[0, 4] == 0 and res.hours[0, 4] == 0
    assert strategy_metrics(res)["p_hit"] == 1.0


def test_coast_scales_income_by_factor():
    s = make(strategy={"kind": "coast", "multiple": 5, "coast_factor": 0.5, "stop": ["Pay"],
                       "expected_real_return": 0.0, "target_year": 19})
    res = simulate(s)
    y = res.strategy_year[0]
    assert y > 0 and res.income[0, y] == 50_000 and res.income[0, y - 1] == 100_000


def test_lean_fire_caps_tier():
    res = simulate(compose(PROFILE, ROOT / "careers" / "quant_finance.yaml",
                           ROOT / "strategies" / "fire_lean.yaml", paths=500))
    assert res.tier.max() <= 1


# --- every file combination loads -----------------------------------------------------------


@pytest.mark.parametrize("career", CAREERS, ids=lambda p: p.stem)
def test_every_career_with_every_addon_composes(career):
    for addon in ADDONS:
        s = compose(PROFILE, career, addon)
        assert from_dict(to_dict(s)) == s, (career.name, addon.name)
    s = compose(PROFILE, career, *(ROOT / "life").glob("*.yaml"),
                ROOT / "strategies" / "coast_fire.yaml", paths=300)
    res = simulate(s)
    assert np.isfinite(res.net_worth).all()


# --- locations and the new career presets ---------------------------------------------

LOCATIONS = sorted((ROOT / "locations").glob("*.yaml"))


@pytest.mark.parametrize("loc", LOCATIONS, ids=lambda p: p.stem)
def test_location_layers_on_every_career(loc):
    for career in CAREERS:
        s = compose(PROFILE, career, loc, paths=50)
        assert simulate(s).net_worth.shape == (50, s.horizon_years + 1)


def test_location_after_partner_keeps_new_state_taxes():
    s = compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml", ROOT / "life" / "partner.yaml",
                ROOT / "locations" / "texas.yaml")
    names = {t.name for t in s.taxes}
    assert not any("California" in n for n in names)
    joint = next(t for t in s.taxes if t.name == "Federal income tax")
    assert joint.deduction == 31500          # the partner's joint federal table survives


def test_no_state_tax_raises_take_home():
    ca = simulate(compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml", paths=200))
    tx = simulate(compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml",
                          ROOT / "locations" / "texas.yaml", paths=200))
    # Pay is 8% lower in Texas, but no state tax: the tax bill drops by more than pay.
    assert tx.tax[:, 5].mean() < 0.8 * ca.tax[:, 5].mean()


def test_cardiology_ep_ladder_reaches_ep_pay_on_some_paths():
    s = compose(PROFILE, ROOT / "careers" / "cardiology_ep.yaml", paths=4000)
    res = simulate(s)
    late = res.income[:, 20]                 # age ~49
    share_ep = np.mean(late > 600_000)
    # 60% admitted x 65% cardiology x 75% EP, about 29%.
    assert 0.22 < share_ep < 0.36


def test_industry_then_founder_has_unpaid_runway():
    s = compose(PROFILE, ROOT / "careers" / "industry_then_founder.yaml", paths=2000)
    res = simulate(s)
    assert np.mean(res.income[:, 5] > 100_000) > 0.95     # still in industry
    assert np.mean(res.income[:, 9] < 50_000) > 0.6        # bootstrapping, living off savings
