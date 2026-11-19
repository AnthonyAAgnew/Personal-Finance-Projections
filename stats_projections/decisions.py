"""Decision points: keep your options open, then switch paths if you can afford to.

A decision point says: follow path A, and at ``year`` look at each simulated
life. Where the rule holds (for example "at least $150k in liquid savings"),
switch to path B from that year on; elsewhere stay on A. Because both paths run
on the same random draws, each life's markets, raises and luck are the same
before and after the choice.

Mechanics: the yearly flows of A (pay, spending, hours, events, loans) are
used before the decision year and on paths that stay; B's flows are used from
the decision year on paths that switch. Pick a B whose own changes start after
the decision (such as "Industry first, then MD / EP" with a decision at 29).
B's events before the decision year are ignored, and A's loans or events after
it are dropped on switching paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Scenario
from .engine import Flows, Result, flows, simulate

RULES = {
    "liquid": "Liquid savings (cash + investments) at least",
    "net_worth": "Net worth at least",
    "always": "Always switch",
}


def _pick(a: np.ndarray | None, b: np.ndarray | None, mask: np.ndarray, fill=0.0) -> np.ndarray:
    shape = (a if a is not None else b).shape
    a = np.full(shape, fill) if a is None else a
    b = np.full(shape, fill) if b is None else b
    return np.where(mask, b, a)


def combine(fa: Flows, fb: Flows, switch: np.ndarray, year: int) -> Flows:
    """Flows of A, replaced by B's on switching paths from ``year`` on."""
    P, T = fa.purchases.shape
    m = switch[:, None] & (np.arange(T)[None, :] >= year)  # (P, T)
    m1 = switch[:, None] & (np.arange(T + 1)[None, :] > year)  # end-of-year columns

    def merged(da: dict, db: dict, fill=0.0) -> dict:
        return {k: _pick(da.get(k), db.get(k), m, fill) for k in dict.fromkeys([*da, *db])}

    off = {k: _pick(fa.off.get(k), fb.off.get(k), m, True).astype(bool)
           for k in dict.fromkeys([*fa.off, *fb.off])}
    return Flows(
        income=merged(fa.income, fb.income),
        taxable={**fb.taxable, **fa.taxable},
        spendable={**fb.spendable, **fa.spendable},
        expenses=merged(fa.expenses, fb.expenses),
        off=off,
        hours=merged(fa.hours, fb.hours),
        purchases=_pick(fa.purchases, fb.purchases, m),
        windfalls=_pick(fa.windfalls, fb.windfalls, m),
        loans_in=_pick(fa.loans_in, fb.loans_in, m),
        loan_payments=_pick(fa.loan_payments, fb.loan_payments, m),
        debt=_pick(fa.debt, fb.debt, m1),
        home_costs=_pick(fa.home_costs, fb.home_costs, m),
        home=_pick(fa.home, fb.home, m1),
    )


@dataclass
class DecisionPoint:
    name: str
    year: int
    switch: np.ndarray  # (paths,) bool
    a: Result
    b: Result
    combined: Result

    @property
    def p_switch(self) -> float:
        return float(self.switch.mean())


def decide(a: Scenario, b: Scenario, year: int, rule: str = "liquid",
           threshold: float = 150_000.0) -> DecisionPoint:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {tuple(RULES)}")
    if (a.paths, a.horizon_years, a.seed) != (b.paths, b.horizon_years, b.seed):
        b = replace(b, paths=a.paths, horizon_years=a.horizon_years, seed=a.seed)
    year = int(np.clip(year, 0, a.horizon_years - 1))
    ra, rb = simulate(a), simulate(b)
    if rule == "always":
        switch = np.ones(a.paths, dtype=bool)
    else:
        state = ra.portfolio[:, year] if rule == "liquid" else ra.net_worth[:, year]
        switch = state >= threshold
    F = combine(flows(a), flows(b), switch, year)
    age = (a.start_age or 0) + year
    label = {"liquid": f"liquid savings >= ${threshold:,.0f}",
             "net_worth": f"net worth >= ${threshold:,.0f}", "always": "always"}[rule]
    name = f"{a.name}, then {b.name} at {age} if {label}"
    combined = simulate(replace(a, name=name), F)
    return DecisionPoint(name, year, switch, ra, rb, combined)
