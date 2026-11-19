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
from .engine import Flows, Result, event_years, flows, simulate

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


# ----------------------------------------------------------------------------
# Decision trees: several chained decision points with richer rules
# ----------------------------------------------------------------------------

#: Things a condition can look at, with the label the app shows.
MEASURES = {
    "always": "Always",
    "liquid": "Liquid savings (cash + investments)",
    "net_worth": "Net worth",
    "pay": "Career pay last year",
    "event": "Event has happened",
    "no_event": "Event has not happened",
}
PAY_STREAM = "Career pay"


@dataclass
class Condition:
    """One test on each simulated life at the decision age.

    ``measure`` is a key of ``MEASURES``. Money measures compare with
    ``value`` (``at_least`` True for >=, False for <). Event measures look for
    ``event`` (by name) on the path the life is following at that point, and
    count it if it started at or before the decision age.
    """

    measure: str = "always"
    value: float = 0.0
    at_least: bool = True
    event: str = ""

    def label(self) -> str:
        if self.measure == "always":
            return "always"
        if self.measure == "event":
            return f"'{self.event}' has happened"
        if self.measure == "no_event":
            return f"'{self.event}' has not happened"
        op = ">=" if self.at_least else "<"
        return f"{MEASURES[self.measure].lower()} {op} ${self.value:,.0f}"


@dataclass
class Step:
    """At ``year``, lives in ``applies_to`` that meet every condition switch to ``to``.

    ``applies_to`` is ``"all"`` or ``(k, "switched")`` / ``(k, "stayed")`` for
    an earlier step ``k`` (0-based), which makes the plan a tree.
    """

    year: int
    to: Scenario
    conditions: list[Condition]
    applies_to: str | tuple[int, str] = "all"


@dataclass
class TreeResult:
    name: str
    start: Result  # always staying on the starting path
    combined: Result  # following the tree
    steps: list[Step]
    switched: list[np.ndarray]  # per step, (paths,) bool: switched at that step
    eligible: list[np.ndarray]  # per step, (paths,) bool: the step applied to this life
    branch: np.ndarray  # (paths,) str label of the route each life took
    on: np.ndarray  # (paths,) index of the path each life ends on (0 = start)


def _measure(c: Condition, year: int, res: Result, F: Flows,
             ev: list[dict[str, np.ndarray]], on: np.ndarray) -> np.ndarray:
    P = res.net_worth.shape[0]
    if c.measure == "always":
        return np.ones(P, dtype=bool)
    if c.measure in ("event", "no_event"):
        happened = np.zeros(P, dtype=bool)
        for k, years in enumerate(ev):
            if c.event in years:
                happened |= (on == k) & (years[c.event] <= year)
        return happened if c.measure == "event" else ~happened
    if c.measure == "liquid":
        x = res.portfolio[:, year] if res.portfolio is not None else res.net_worth[:, year]
    elif c.measure == "net_worth":
        x = res.net_worth[:, year]
    elif c.measure == "pay":
        pay = F.income.get(PAY_STREAM, F.total_income)
        x = pay[:, max(year - 1, 0)]
    else:
        raise ValueError(f"unknown measure {c.measure!r}; use one of {tuple(MEASURES)}")
    return x >= c.value if c.at_least else x < c.value


def decide_tree(start: Scenario, steps: list[Step]) -> TreeResult:
    """Follow ``start`` and apply each step in order of listing.

    Steps run on the result so far, so a later step sees savings and pay
    produced by earlier switches. All paths share one set of random draws.
    """
    if not steps:
        raise ValueError("add at least one step")
    T = start.horizon_years
    scen = [start]
    fixed = []
    for st_ in steps:
        to = st_.to
        if (to.paths, to.horizon_years, to.seed) != (start.paths, T, start.seed):
            to = replace(to, paths=start.paths, horizon_years=T, seed=start.seed)
        fixed.append(replace(st_, to=to, year=int(np.clip(st_.year, 0, T - 1))))
        scen.append(to)
    for i, st_ in enumerate(fixed):
        if st_.applies_to != "all":
            k, how = st_.applies_to
            if not 0 <= k < i or how not in ("switched", "stayed"):
                raise ValueError(f"step {i + 1}: can only depend on an earlier step")

    ev = [event_years(s) for s in scen]
    F = flows(start)
    res0 = simulate(start)
    res = res0
    on = np.zeros(start.paths, dtype=np.int64)
    route = [[] for _ in range(start.paths)]
    a0 = start.start_age or 0
    switched, eligible = [], []
    for i, st_ in enumerate(fixed):
        if st_.applies_to == "all":
            elig = np.ones(start.paths, dtype=bool)
        else:
            k, how = st_.applies_to
            elig = switched[k] if how == "switched" else eligible[k] & ~switched[k]
        sw = elig.copy()
        for c in st_.conditions or [Condition()]:
            sw &= _measure(c, st_.year, res, F, ev, on)
        switched.append(sw)
        eligible.append(elig)
        if sw.any():
            F = combine(F, flows(st_.to), sw, st_.year)
            on = np.where(sw, i + 1, on)
            for p in np.flatnonzero(sw):
                route[p].append(f"{st_.to.name} at {a0 + st_.year}")
            res = simulate(replace(start, name="tree"), F)

    branch = np.array([" → ".join(r) if r else f"Stayed on {start.name}" for r in route],
                      dtype=object)
    name = f"Decision tree from {start.name}"
    combined = simulate(replace(start, name=name), F)
    return TreeResult(name, res0, combined, fixed, switched, eligible, branch, on)


def branch_summary(tree: TreeResult, year: int) -> list[dict]:
    """One row per route: share of lives and outcomes at ``year`` (column)."""
    nw = tree.combined.net_worth[:, year]
    stay = tree.start.net_worth[:, year]
    rows = []
    for b in dict.fromkeys(tree.branch):
        m = tree.branch == b
        rows.append({"Route": b, "Share of lives": float(m.mean()),
                     "Median": float(np.median(nw[m])),
                     "P10": float(np.percentile(nw[m], 10)),
                     "P90": float(np.percentile(nw[m], 90)),
                     "Cash crunch": float(tree.combined.failed[m].mean()),
                     "Median gain vs staying": float(np.median(nw[m] - stay[m]))})
    rows.sort(key=lambda r: -r["Share of lives"])
    return rows
