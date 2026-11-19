from pathlib import Path

import pytest

from stats_projections import Steps, analyze, format_sensitivity, load_scenario, plot_tornado
from stats_projections.config import from_dict
from stats_projections.sensitivity import knobs

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


def scenario(**over):
    raw = {
        "name": "s",
        "horizon_years": 30,
        "paths": 2000,
        "seed": 3,
        "portfolio": {
            "starting_balance": 1000,
            "returns": {"model": "normal", "mean": 0.06, "stdev": 0.18},
            "inflation": {"mean": 0.02, "stdev": 0.01},
        },
        "income": [{"name": "Pay", "amount": 100, "growth": {"mean": 0.0, "stdev": 0.05},
                    "loss_probability": 0.05}],
        "expenses": [{"name": "Spend", "amount": 97}],
        "events": [
            {"name": "Car", "type": "purchase", "amount": 500, "year": 5},
            {"name": "Car upkeep", "type": "purchase", "amount": 50,
             "same_time_as": "Car", "offset": 1},
        ],
    }
    raw.update(over)
    return from_dict(raw)


def labels(s, steps=Steps()):
    return [k.label for k in knobs(s, steps)]


def test_knob_list_covers_inputs():
    got = labels(scenario())
    for want in ["Return mean", "Return volatility", "Inflation mean", "Starting balance",
                 "Pay: amount", "Pay: growth", "Pay: job-gap chance", "Spend: amount",
                 "Car: amount", "Car: timing (+ linked)", "Car upkeep: amount"]:
        assert want in got
    # Linked events move with their anchor, so they get no timing knob of their own.
    assert "Car upkeep: timing" not in got


def test_bootstrap_has_no_return_knobs(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("year,return,inflation\n1,0.1,0.02\n2,-0.1,0.03\n")
    s = scenario(portfolio={"starting_balance": 1000,
                            "returns": {"model": "bootstrap", "data": str(csv)}})
    assert not any(l.startswith(("Return", "Inflation")) for l in labels(s))


def test_directions_make_sense():
    sa = analyze(scenario())
    by = {r.knob.label: r for r in sa.rows}
    base = sa.base.p_fail
    assert 0.05 < base < 0.95  # the test scenario is actually uncertain
    assert by["Spend: amount"].p_high > base > by["Spend: amount"].p_low
    assert by["Pay: amount"].p_high < base < by["Pay: amount"].p_low
    assert by["Return mean"].p_high <= base <= by["Return mean"].p_low
    assert by["Car: amount"].p_high >= base >= by["Car: amount"].p_low


def test_rows_sorted_by_swing_and_spending_dominates():
    sa = analyze(scenario())
    swings = [r.swing for r in sa.rows]
    assert swings == sorted(swings, reverse=True)
    # Saving 3 of 100 a year, a 10% change in spending or pay matters most.
    assert sa.rows[0].knob.label in ("Spend: amount", "Pay: amount")


def test_steps_are_configurable():
    k = {k.label: k for k in knobs(scenario(), Steps(money=0.5, years=0))}
    assert k["Spend: amount"].low.stream("Spend").amount == pytest.approx(48.5)
    assert not any("timing" in label for label in k)


def test_unchanged_inputs_keep_draws():
    """Nudging one stream leaves the market draws identical (paired comparison)."""
    from stats_projections import simulate

    s = scenario()
    k = {k.label: k for k in knobs(s)}["Spend: amount"]
    import numpy as np

    np.testing.assert_array_equal(simulate(s).real_returns, simulate(k.high).real_returns)


def test_report_and_chart_on_example(tmp_path):
    s = load_scenario(SCENARIOS / "buy_home.yaml", paths=1000)
    sa = analyze(s)
    text = format_sensitivity(sa, top=5)
    assert "Sensitivity: Buy a home" in text and "smaller inputs not shown" in text
    out = plot_tornado(sa, tmp_path / "t.png", top=8)
    assert out.exists() and out.stat().st_size > 10_000
