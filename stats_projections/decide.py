"""Weighted decision scores and a Pareto view across options.

Money is only part of a career choice. This module turns each simulated
option into a handful of criteria, some financial (wealth, downside, age of
financial independence) and some about life (hours, years before real pay,
autonomy, satisfaction, burnout). It scores them on a common 0-100 scale with
weights you choose, and finds the options no other option beats on both the
financial and the life score (the Pareto frontier).

Scores are relative: 0 is the worst option in the comparison on that
criterion, 100 the best. Adding or removing options changes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Result
from .report import hours_metrics


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    higher_is_better: bool
    group: str  # "financial" or "life"
    unit: str  # "money", "age", "hours", "score", "share"
    help: str


CRITERIA = (
    Criterion("wealth", "Net worth at 65 (median)", True, "financial", "money",
              "Typical outcome: the median path's net worth at 65."),
    Criterion("downside", "Net worth at 65 (P10)", True, "financial", "money",
              "Bad-luck outcome: 1 path in 10 ends below this."),
    Criterion("fi_age", "Financial independence age (median)", False, "financial", "age",
              "First age the portfolio reaches 25x that year's spending "
              "(work becomes optional). Paths that never get there count as the plan's end."),
    Criterion("training", "Years working for under $100k (median)", False, "life", "years",
              "Years of school, training, postdoc or bootstrapping: working (hours > 0) "
              "but earning under $100k pre-tax. Includes the Ph.D. years for everyone."),
    Criterion("hours", "Lifetime work hours (median)", False, "life", "hours",
              "Total hours worked from now to retirement."),
    Criterion("autonomy", "Autonomy (O*NET, 0-100)", True, "life", "score",
              "O*NET 'Freedom to Make Decisions' for the career's main job."),
    Criterion("satisfaction", "Satisfied / would choose again", True, "life", "share",
              "Share of people in the job who are satisfied or would choose it again (surveys)."),
    Criterion("burnout", "Burnout rate", False, "life", "share",
              "Share of people in the job reporting burnout (surveys; definitions vary)."),
)
BY_KEY = {c.key: c for c in CRITERIA}

DEFAULT_WEIGHTS = {
    "wealth": 5, "downside": 3, "fi_age": 4, "training": 2,
    "hours": 3, "autonomy": 3, "satisfaction": 4, "burnout": 3,
}


def _first_year(mask: np.ndarray) -> np.ndarray:
    """First column where mask is True on each row, or -1."""
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1), -1)


def _median_age(years: np.ndarray, a0: int, end_age: float) -> float:
    ages = np.where(years >= 0, years + a0, end_age).astype(float)
    return float(np.median(ages))


def criteria_values(res: Result, at_age: int = 65, fi_multiple: float = 25.0,
                    pay_level: float = 100_000.0) -> dict[str, float]:
    """The raw value of every criterion for one simulated option."""
    s = res.scenario
    a0 = s.start_age or 0
    end_age = a0 + s.horizon_years
    col = int(np.clip(at_age - a0, 0, s.horizon_years))
    nw = res.net_worth[:, col]
    out = {
        "wealth": float(np.median(nw)),
        "downside": float(np.percentile(nw, 10)),
    }
    if res.portfolio is not None and res.spending is not None:
        w = res.portfolio[:, 1:]  # balance at the end of each year
        # Judge FI on the budget your pay sets, not on extra spending the
        # portfolio itself funds (which would make FI a moving target).
        sp = res.core_spending if res.core_spending is not None else res.spending
        fi = _first_year((w >= fi_multiple * sp) & (sp > 0))
        out["fi_age"] = _median_age(fi + 1, a0, end_age)
    else:
        out["fi_age"] = np.nan
    if res.income is not None and res.hours is not None:
        low = (res.hours > 0) & (res.income < pay_level)
        out["training"] = float(np.median(low.sum(axis=1)))
    else:
        out["training"] = np.nan
    hm = hours_metrics(res)
    out["hours"] = hm.lifetime_hours if hm else np.nan
    q = s.quality
    out["autonomy"] = np.nan if q is None or q.autonomy is None else q.autonomy
    out["satisfaction"] = np.nan if q is None or q.satisfaction is None else q.satisfaction
    out["burnout"] = np.nan if q is None or q.burnout is None else q.burnout
    return out


def normalize(values: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per criterion, map options to 0 (worst) .. 1 (best). Missing values get 0.5."""
    names = list(values)
    out: dict[str, dict[str, float]] = {n: {} for n in names}
    for c in CRITERIA:
        v = np.array([values[n].get(c.key, np.nan) for n in names], dtype=float)
        ok = np.isfinite(v)
        if ok.sum() == 0:
            norm = np.full(len(v), 0.5)
        else:
            lo, hi = v[ok].min(), v[ok].max()
            norm = np.full(len(v), 0.5) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
            if not c.higher_is_better:
                norm = 1 - norm
            norm = np.where(ok, norm, 0.5)
        for n, x in zip(names, norm):
            out[n][c.key] = float(x)
    return out


def weighted(norm: dict[str, float], weights: dict[str, float], group: str | None = None) -> float:
    """0-100 weighted average of normalized criteria (optionally one group)."""
    keys = [c.key for c in CRITERIA if group is None or c.group == group]
    w = np.array([max(weights.get(k, 0.0), 0.0) for k in keys], dtype=float)
    if w.sum() == 0:
        return np.nan
    x = np.array([norm[k] for k in keys])
    return float(100 * (w * x).sum() / w.sum())


@dataclass(frozen=True)
class Scored:
    name: str
    values: dict[str, float]
    normalized: dict[str, float]
    total: float
    financial: float
    life: float


def score(results: list[Result], weights: dict[str, float] | None = None,
          at_age: int = 65, fi_multiple: float = 25.0) -> list[Scored]:
    """Score every option, best total first."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    values = {r.scenario.name: criteria_values(r, at_age, fi_multiple) for r in results}
    norm = normalize(values)
    out = [Scored(n, values[n], norm[n], weighted(norm[n], weights),
                  weighted(norm[n], weights, "financial"), weighted(norm[n], weights, "life"))
           for n in values]
    return sorted(out, key=lambda s: -np.nan_to_num(s.total, nan=-1))


def pareto(points: list[tuple[float, float]]) -> list[bool]:
    """True for points that no other point beats on both axes (higher is better).

    A point is dominated when another is at least as good on both and strictly
    better on one.
    """
    pts = np.asarray(points, dtype=float)
    keep = []
    for i, (x, y) in enumerate(pts):
        others = np.delete(pts, i, axis=0)
        dominated = bool(np.any((others[:, 0] >= x) & (others[:, 1] >= y)
                                & ((others[:, 0] > x) | (others[:, 1] > y))))
        keep.append(not dominated)
    return keep
