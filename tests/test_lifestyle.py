from pathlib import Path

import numpy as np
import pytest

from stats_projections import ScenarioError, simulate
from stats_projections.config import compose, from_dict
from stats_projections.engine import CREEP, progressive_tax
from stats_projections.report import (gross_for_take_home, income_milestones,
                                      net_worth_milestones, tier_shares)

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profiles" / "my_profile.yaml"


def make(**over):
    raw = {
        "name": "t", "horizon_years": 10, "paths": 50, "seed": 1,
        "portfolio": {"starting_balance": 0,
                      "returns": {"mean": 0.0, "stdev": 0.0},
                      "inflation": {"mean": 0.0, "stdev": 0.0}},
        "income": [{"name": "Pay", "amount": 100_000}],
        "expenses": [{"name": "Rent", "monthly": 1000}, {"name": "Fun", "monthly": 100}],
        "lifestyle": {"creep": 0.5, "withdrawal_rate": 0.0, "smoothing_years": 1,
                      "tiers": [{"name": "Nice", "take_home": 80_000,
                                 "monthly": {"Rent": 2000, "Fun": 500}}]},
    }
    raw.update(over)
    return from_dict(raw)


def test_monthly_amounts_are_yearly_times_12():
    s = make(lifestyle=None)
    assert s.stream("Rent").amount == 12_000
    with pytest.raises(ScenarioError, match="exactly one"):
        make(expenses=[{"name": "Rent", "monthly": 1, "amount": 12}], lifestyle=None)


def test_tier_budget_plus_creep_by_hand():
    res = simulate(make())
    # Affordable 100k >= 80k: tier budget 12*(2000+500)=30k, creep 0.5*(100k-80k)=10k.
    np.testing.assert_allclose(res.cash_flow[0], 100_000 - 30_000 - 10_000)
    assert (res.tier == 1).all()
    np.testing.assert_allclose(res.flow_means["spending"][CREEP], 10_000)


def test_below_first_tier_uses_file_budget_and_no_creep():
    res = simulate(make(income=[{"name": "Pay", "amount": 50_000}]))
    np.testing.assert_allclose(res.cash_flow[0], 50_000 - 13_200)
    assert (res.tier == 0).all()


def test_portfolio_draw_counts_toward_affordable_income():
    s = make(income=[{"name": "Pay", "amount": 50_000}],
             portfolio={"starting_balance": 1_000_000,
                        "returns": {"mean": 0.0, "stdev": 0.0},
                        "inflation": {"mean": 0.0, "stdev": 0.0}},
             lifestyle={"creep": 0.0, "withdrawal_rate": 0.04, "smoothing_years": 1,
                        "tiers": [{"name": "Nice", "take_home": 80_000, "monthly": {"Rent": 2000}}]})
    res = simulate(s)  # 50k + 4% x 1M = 90k >= 80k
    assert res.tier[0, 0] == 1


def test_paused_category_stays_paused_in_a_tier():
    s = make(events=[{"name": "Away", "type": "set_amount", "target": ["Rent", "Fun"],
                      "amount": 0, "year": 2, "duration": 2}])
    res = simulate(s)
    spend = res.flow_means["spending"]
    assert spend["Rent"][2] == 0 and spend["Fun"][3] == 0 and spend[CREEP][2] == 0
    assert spend["Rent"][4] == 24_000


def test_target_list_expands_to_linked_events():
    s = make(events=[{"name": "Away", "type": "set_amount", "target": ["Rent", "Fun"],
                      "amount": 0, "year": 2, "probability": 0.5}])
    names = {e.name: e for e in s.events}
    assert names["Away"].target == "Rent" and names["Away (Fun)"].same_time_as == "Away"


def test_tier_categories_must_exist():
    with pytest.raises(ScenarioError, match="unknown expense categories"):
        make(lifestyle={"tiers": [{"name": "x", "take_home": 1, "monthly": {"Yacht": 9}}]})


def test_cash_flow_means_balance():
    res = simulate(compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml", paths=500))
    fm = res.flow_means
    inflow = sum(fm["income"].values()) + fm["other"]["Windfalls (after tax)"] + fm["other"]["Loans received"]
    outflow = (sum(fm["tax"].values()) + sum(fm["spending"].values())
               + fm["other"]["One-time purchases"] + fm["other"]["Loan payments"] + fm["saved"])
    np.testing.assert_allclose(inflow, outflow, rtol=1e-9, atol=1e-6)


def test_gross_for_take_home_inverts_taxes():
    s = compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml")
    g = gross_for_take_home(s, 200_000)
    assert g - progressive_tax(np.array([g]), s.taxes)[0] == pytest.approx(200_000, abs=1)


def test_milestones():
    res = simulate(make(horizon_years=10, lifestyle=None))
    inc = income_milestones(res, (50_000, 150_000))
    assert inc[0].p_ever == 1 and inc[0].median_age == 0 and inc[1].p_ever == 0
    nw = net_worth_milestones(res, (400_000,), withdrawal_rate=0.04)[0]
    # Saves 100k - 13.2k = 86.8k a year: 400k is passed at the end of year 4 (index 4).
    assert nw.median_age == 4 and nw.dividend == 16_000 and nw.by_age[-1] == 1


def test_profile_tiers_rise_with_career():
    res = simulate(compose(PROFILE, ROOT / "careers" / "device_sales.yaml", paths=2000))
    sh = tier_shares(res)
    assert sh["Base budget"][0] == 1.0  # Ph.D. years
    assert sh["Base budget"][15] < 0.2  # well into a sales career
