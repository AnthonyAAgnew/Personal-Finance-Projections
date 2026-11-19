import numpy as np
import pytest

from stats_projections.decisions import Condition, Step, branch_summary, decide, decide_tree

from test_new_features import make


def pay(amount, **over):
    return make(income=[{"name": "Career pay", "amount": amount}], **over)


def test_single_step_matches_decision_point():
    a, b = pay(100_000), pay(50_000)
    old = decide(a, b, 5, "always")
    new = decide_tree(a, [Step(5, b, [Condition("always")])])
    np.testing.assert_allclose(new.combined.net_worth, old.combined.net_worth)
    assert set(new.branch) == {"t at 5"}


def test_money_rules_split_lives():
    a, b = pay(100_000), pay(50_000)
    never = decide_tree(a, [Step(5, b, [Condition("liquid", 1e12)])])
    np.testing.assert_allclose(never.combined.net_worth, never.start.net_worth)
    assert set(never.branch) == {"Stayed on t"}
    below = decide_tree(a, [Step(5, b, [Condition("pay", 80_000, at_least=False)])])
    assert not below.switched[0].any()                  # pay is 100k, not below 80k
    both = decide_tree(a, [Step(5, b, [Condition("pay", 80_000), Condition("liquid", 1e12)])])
    assert not both.switched[0].any()                   # "and": second fails


def test_event_rule_and_chained_branches():
    # Half the lives get an offer at year 3 that raises pay.
    a = pay(100_000, events=[{"name": "Offer", "type": "set_amount", "target": "Career pay",
                              "amount": 200_000, "year": 3, "probability": 0.5}])
    b, c = pay(50_000), pay(70_000)
    tree = decide_tree(a, [
        Step(4, b, [Condition("no_event", event="Offer")]),
        Step(8, c, [Condition("always")], applies_to=(0, "switched")),
    ])
    offer = tree.switched[0]
    assert 0.2 < (~offer).mean() < 0.8
    np.testing.assert_array_equal(tree.switched[1], offer)   # only lives that switched first
    np.testing.assert_allclose(tree.combined.income[~offer, 3:], 200_000)
    np.testing.assert_allclose(tree.combined.income[offer, 4:8], 50_000)
    np.testing.assert_allclose(tree.combined.income[offer, 8:], 70_000)
    rows = branch_summary(tree, 20)
    assert {r["Route"] for r in rows} == {"Stayed on t", "t at 4 → t at 8"}
    assert sum(r["Share of lives"] for r in rows) == pytest.approx(1)


def test_later_step_sees_event_on_new_path():
    a = pay(100_000)
    b = pay(60_000, events=[{"name": "Promo", "type": "set_amount", "target": "Career pay",
                             "amount": 90_000, "year": 6}])
    tree = decide_tree(a, [Step(2, b, [Condition("always")]),
                           Step(10, a, [Condition("event", event="Promo")])])
    assert tree.switched[1].all()


def test_bad_dependency_is_rejected():
    a = pay(100_000)
    with pytest.raises(ValueError):
        decide_tree(a, [Step(3, a, [], applies_to=(0, "switched"))])
    with pytest.raises(ValueError):
        decide_tree(a, [])
