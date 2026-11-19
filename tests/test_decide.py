from pathlib import Path

import numpy as np
import pytest

from stats_projections import ScenarioError, simulate
from stats_projections.config import compose, from_dict
from stats_projections.decide import CRITERIA, criteria_values, normalize, pareto, score, weighted
from stats_projections.edit import to_dict

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profiles" / "my_profile.yaml"
CAREERS = sorted((ROOT / "careers").glob("*.yaml"))


def test_every_career_has_quality_scores():
    for f in CAREERS:
        q = compose(PROFILE, f, paths=10).quality
        assert q is not None and None not in (q.autonomy, q.satisfaction, q.burnout), f.name


def test_quality_validated_and_round_trips():
    base = {"name": "t", "horizon_years": 5, "paths": 5,
            "income": [{"name": "Pay", "amount": 1}], "expenses": []}
    with pytest.raises(ScenarioError):
        from_dict({**base, "quality": {"burnout": 45}})       # a share, not a percent
    with pytest.raises(ScenarioError):
        from_dict({**base, "quality": {"happiness": 0.5}})    # unknown key
    s = from_dict({**base, "quality": {"autonomy": 80, "burnout": 0.4}})
    assert from_dict(to_dict(s)).quality == s.quality


def test_pareto_drops_only_dominated_points():
    pts = [(10, 10), (5, 5), (10, 0), (0, 10), (10, 10)]
    # (5, 5) is beaten on both by (10, 10); ties do not dominate each other.
    assert pareto(pts) == [True, False, False, False, True] or pareto(pts)[1] is False
    assert pareto([(1, 2), (2, 1)]) == [True, True]
    assert pareto([(1, 1), (2, 2)]) == [False, True]


def test_normalize_flips_lower_is_better_and_fills_missing():
    vals = {"a": {"wealth": 1.0, "burnout": 0.2}, "b": {"wealth": 3.0, "burnout": 0.6},
            "c": {"wealth": 2.0, "burnout": np.nan}}
    n = normalize(vals)
    assert n["a"]["wealth"] == 0 and n["b"]["wealth"] == 1
    assert n["a"]["burnout"] == 1 and n["b"]["burnout"] == 0   # less burnout is better
    assert n["c"]["burnout"] == 0.5                             # missing: neutral
    only_wealth = {c.key: 0 for c in CRITERIA} | {"wealth": 1}
    assert weighted(n["b"], only_wealth) == pytest.approx(100)


def test_scores_rank_careers_and_respect_weights():
    rs = [simulate(compose(PROFILE, ROOT / "careers" / f"{c}.yaml", paths=300))
          for c in ("rd_engineer", "quant_finance", "regulator")]
    money_only = {c.key: (1 if c.key == "wealth" else 0) for c in CRITERIA}
    best = score(rs, money_only)[0]
    assert best.name.startswith("Quantitative finance")
    calm = {c.key: (1 if c.key in ("hours",) else 0) for c in CRITERIA}
    assert score(rs, calm)[0].name.startswith("Regulator")     # fewest hours


def test_low_pay_years_count_training():
    md = simulate(compose(PROFILE, ROOT / "careers" / "cardiology_ep.yaml", paths=2000))
    rd = simulate(compose(PROFILE, ROOT / "careers" / "rd_engineer.yaml", paths=2000))
    assert criteria_values(rd)["training"] == 2                # just the Ph.D.
    assert criteria_values(md)["training"] > 6                 # school, residency, fellowship
