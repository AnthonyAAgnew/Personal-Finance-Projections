"""Break-even finder: what would have to be true for option A to match option B?

Pick one money input of option A (EP attending pay, exit price, tuition...)
and a yardstick (median net worth at 65, say). The finder searches for the
value of that input at which A's yardstick equals B's, re-simulating A on
the same random draws each time (bisection). Answers questions like "at what
EP pay does medicine catch up with industry?".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .config import Scenario
from .engine import Result, event_years, simulate


@dataclass(frozen=True)
class Lever:
    label: str
    value: float  # as written in the files
    apply: Callable[[float], Scenario]
    event: str | None = None  # the event it belongs to, if any


def levers(s: Scenario) -> list[Lever]:
    """Money inputs that can be solved for."""
    out: list[Lever] = []
    for x in s.income:
        if x.amount > 0:
            out.append(Lever(f"Income: {x.name} (starting amount)", x.amount,
                             lambda v, n=x.name: replace(s, income=tuple(
                                 replace(i, amount=v) if i.name == n else i for i in s.income))))
    for e in s.events:
        if e.equity is not None:
            out.append(Lever(f"Event: {e.name} (sale price, median)", e.equity.sale_median,
                             lambda v, n=e.name: replace(s, events=tuple(
                                 replace(x, equity=replace(x.equity, sale_median=v))
                                 if x.name == n else x for x in s.events)), e.name))
        elif e.amount > 0 and e.type in ("set_amount", "windfall", "purchase", "loan", "home"):
            kind = {"set_amount": "pay or cost from then on", "windfall": "windfall",
                    "purchase": "cost", "loan": "loan", "home": "price"}[e.type]
            out.append(Lever(f"Event: {e.name} ({kind})", e.amount,
                             lambda v, n=e.name: replace(s, events=tuple(
                                 replace(x, amount=v) if x.name == n else x for x in s.events)),
                             e.name))
    return out


METRICS: dict[str, Callable[[Result, int], float]] = {
    "Median net worth": lambda r, col: float(np.median(r.net_worth[:, col])),
    "P10 net worth (bad luck)": lambda r, col: float(np.percentile(r.net_worth[:, col], 10)),
    "Mean net worth": lambda r, col: float(np.mean(r.net_worth[:, col])),
}


@dataclass
class BreakEven:
    lever: str
    base_value: float
    target: float  # option B's yardstick
    base_metric: float  # option A's yardstick as written
    value: float | None  # the break-even input, None if not in range
    note: str
    curve: list[tuple[float, float]]  # (input, A's yardstick) points tried
    reach: float = 1.0  # share of A's paths where the input applies at all


def solve(a: Scenario, b: Scenario, lever: Lever, metric: str = "Median net worth",
          at_age: int = 65, paths: int = 2000, lo: float | None = None,
          hi: float | None = None, iters: int = 14) -> BreakEven:
    a0 = a.start_age or 0
    col = int(np.clip(at_age - a0, 0, a.horizon_years))
    f = METRICS[metric]
    target = f(simulate(replace(b, paths=paths)), col)
    curve: list[tuple[float, float]] = []

    def at(v: float) -> float:
        m = f(simulate(replace(lever.apply(v), paths=paths)), col)
        curve.append((v, m))
        return m

    base_metric = at(lever.value)
    reach = 1.0
    if lever.event:
        reach = float((event_years(a)[lever.event] < a.horizon_years).mean())
    lo = 0.0 if lo is None else lo
    hi = 5.0 * lever.value if hi is None else hi
    m_lo, m_hi = at(lo), at(hi)
    if (m_lo - target) * (m_hi - target) > 0:
        side = "above" if m_lo > target else "below"
        return BreakEven(lever.label, lever.value, target, base_metric, None,
                         f"Option A stays {side} option B for every value from {lo:,.0f} to "
                         f"{hi:,.0f}: no break-even in that range.", sorted(curve), reach)
    for _ in range(iters):
        mid = (lo + hi) / 2
        m_mid = at(mid)
        if (m_lo - target) * (m_mid - target) <= 0:
            hi, m_hi = mid, m_mid
        else:
            lo, m_lo = mid, m_mid
    value = (lo + hi) / 2
    return BreakEven(lever.label, lever.value, target, base_metric, value,
                     "Found by bisection on the same random draws.", sorted(curve), reach)
