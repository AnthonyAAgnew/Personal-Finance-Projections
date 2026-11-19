"""One-at-a-time sensitivity analysis: which input drives failure risk most?

Each "knob" nudges one input down and up while everything else stays fixed,
and the scenario is re-simulated with the same random draws. Because the
draws are shared (common random numbers), the change in P(fail) reflects the
input, not sampling noise, and comes with a tight paired confidence interval.

Default nudges (change them with ``Steps``):

* money amounts (starting balance, income, expenses, events): +/- 10%
* rates (return mean, inflation mean, stream growth): +/- 1 point
  (0.5 point for stream growth)
* return volatility: +/- 3 points
* job-gap probability: +/- 2 points
* event timing: +/- 2 years. Linked events (``same_time_as``) move with it.

Read the ranking as "what to estimate carefully", not as a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .config import Event, Scenario, Stream
from .engine import Result, simulate
from .report import Difference, compare


@dataclass(frozen=True)
class Steps:
    money: float = 0.10  # relative
    rate: float = 0.01  # absolute, return mean and inflation mean
    growth: float = 0.005  # absolute, stream growth mean
    volatility: float = 0.03  # absolute, return stdev
    probability: float = 0.02  # absolute, job-gap probability
    years: int = 2


@dataclass(frozen=True)
class Knob:
    label: str
    group: str  # Market | Starting point | Income | Expenses | Events
    low_desc: str
    high_desc: str
    low: Scenario
    high: Scenario


@dataclass
class KnobResult:
    knob: Knob
    low: Difference  # versus the unchanged scenario
    high: Difference
    p_low: float
    p_high: float

    @property
    def swing(self) -> float:
        """Largest move in failure probability, in either direction."""
        return max(abs(self.low.delta), abs(self.high.delta))


@dataclass
class Sensitivity:
    base: Result
    rows: list[KnobResult]  # sorted, biggest swing first
    steps: Steps


# ----------------------------------------------------------------------------
# Building knobs
# ----------------------------------------------------------------------------


def _money(x: float) -> str:
    return f"${x / 1e3:,.1f}k" if abs(x) >= 1e3 else f"${x:,.0f}"


def _pts(x: float) -> str:
    return f"{100 * x:.1f}%"


def _with_stream(s: Scenario, name: str, **changes) -> Scenario:
    def fix(streams: tuple[Stream, ...]) -> tuple[Stream, ...]:
        return tuple(replace(x, **changes) if x.name == name else x for x in streams)

    return replace(s, income=fix(s.income), expenses=fix(s.expenses))


def _with_event(s: Scenario, name: str, **changes) -> Scenario:
    return replace(
        s, events=tuple(replace(e, **changes) if e.name == name else e for e in s.events)
    )


def knobs(s: Scenario, steps: Steps = Steps()) -> list[Knob]:
    out: list[Knob] = []
    m = steps.money

    def pair(label, group, make: Callable[[float], Scenario], fmt, lo, hi):
        if lo == hi:
            return
        out.append(Knob(label, group, fmt(lo), fmt(hi), make(lo), make(hi)))

    # Market. Bootstrap returns come straight from history, so there is no
    # mean or stdev to nudge.
    if s.returns.model != "bootstrap":
        r = s.returns
        pair("Return mean", "Market",
             lambda v: replace(s, returns=replace(r, mean=v)), _pts,
             r.mean - steps.rate, r.mean + steps.rate)
        pair("Return volatility", "Market",
             lambda v: replace(s, returns=replace(r, stdev=v)), _pts,
             max(0.0, r.stdev - steps.volatility), r.stdev + steps.volatility)
        i = s.inflation
        pair("Inflation mean", "Market",
             lambda v: replace(s, inflation=replace(i, mean=v)), _pts,
             i.mean - steps.rate, i.mean + steps.rate)

    ls = s.lifestyle
    if ls is not None and ls.tiers:
        pair("Lifestyle creep", "Lifestyle",
             lambda v: replace(s, lifestyle=replace(ls, creep=v)), _pts,
             max(0.0, ls.creep - 0.10), min(1.0, ls.creep + 0.10))
        pair("Tier budgets (all)", "Lifestyle",
             lambda v: replace(s, lifestyle=replace(ls, tiers=tuple(
                 replace(t, monthly=tuple((c, a * v) for c, a in t.monthly)) for t in ls.tiers))),
             lambda v: f"x{v:.2f}", 1 - m, 1 + m)
        pair("Withdrawal rate", "Lifestyle",
             lambda v: replace(s, lifestyle=replace(ls, withdrawal_rate=v)), _pts,
             max(0.0, ls.withdrawal_rate - 0.005), ls.withdrawal_rate + 0.005)

    if s.starting_balance:
        b = s.starting_balance
        pair("Starting balance", "Starting point",
             lambda v: replace(s, starting_balance=v), _money, b * (1 - m), b * (1 + m))

    for group, streams in (("Income", s.income), ("Expenses", s.expenses)):
        for st in streams:
            if st.amount == 0:
                continue
            n = st.name
            pair(f"{n}: amount", group,
                 lambda v, n=n: _with_stream(s, n, amount=v), _money,
                 st.amount * (1 - m), st.amount * (1 + m))
            g = st.growth
            pair(f"{n}: growth", group,
                 lambda v, n=n, g=g: _with_stream(s, n, growth=replace(g, mean=v)), _pts,
                 g.mean - steps.growth, g.mean + steps.growth)
            if st.loss_probability > 0:
                p = st.loss_probability
                pair(f"{n}: job-gap chance", group,
                     lambda v, n=n: _with_stream(s, n, loss_probability=v), _pts,
                     max(0.0, p - steps.probability), min(1.0, p + steps.probability))

    for e in s.events:
        n = e.name
        if e.amount > 0:
            pair(f"{n}: amount", "Events",
                 lambda v, n=n: _with_event(s, n, amount=v), _money,
                 e.amount * (1 - m), e.amount * (1 + m))
        if not e.same_time_as and steps.years:
            linked = " (+ linked)" if any(x.same_time_as == n for x in s.events) else ""
            pair(f"{n}: timing{linked}", "Events",
                 lambda v, n=n: _with_event(s, n, year=int(v)),
                 (lambda v: f"age {s.start_age + int(v)}") if s.start_age is not None
                 else (lambda v: f"year {int(v)}"),
                 max(0, e.year - steps.years), e.year + steps.years)
    return out


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------


def analyze(
    s: Scenario,
    steps: Steps = Steps(),
    base: Result | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Sensitivity:
    base = base or simulate(s)
    ks = knobs(s, steps)
    rows = []
    for i, k in enumerate(ks):
        lo, hi = simulate(k.low), simulate(k.high)
        rows.append(KnobResult(k, compare(base, lo), compare(base, hi), lo.p_fail, hi.p_fail))
        if progress:
            progress(i + 1, len(ks))
    rows.sort(key=lambda r: r.swing, reverse=True)
    return Sensitivity(base, rows, steps)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def format_sensitivity(sens: Sensitivity, top: int | None = None) -> str:
    s = sens.base.scenario
    rows = sens.rows[:top] if top else sens.rows
    lines = [
        f"Sensitivity: {s.name}  (P(fail) {100 * sens.base.p_fail:.1f}% as entered)",
        "Each input nudged down and up alone; same random draws, so changes are paired.",
        "",
    ]
    header = f"{'Input':<42}{'Low':>11}{'P(fail)':>9}{'High':>11}{'P(fail)':>9}{'Swing':>8}"
    lines += [header, "-" * len(header)]
    for r in rows:
        k = r.knob
        lines.append(
            f"{k.label[:41]:<42}{k.low_desc:>11}{100 * r.p_low:>8.1f}%"
            f"{k.high_desc:>11}{100 * r.p_high:>8.1f}%{100 * r.swing:>7.1f}"
        )
    if top and len(sens.rows) > top:
        lines.append(f"... {len(sens.rows) - top} smaller inputs not shown")
    flat = [r for r in sens.rows
            if max(abs(r.low.ci[0]), abs(r.low.ci[1]), abs(r.high.ci[0]), abs(r.high.ci[1])) < 0.005]
    lines += ["", "Swing = biggest change in P(fail), in percentage points."]
    if flat:
        lines.append(f"{len(flat)} input(s) moved P(fail) by under 0.5 points either way.")
    return "\n".join(lines)


def plot_tornado(sens: Sensitivity, path, top: int = 15):
    """Tornado chart: bars run from P(fail) at the low value to the high value."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = sens.rows[:top][::-1]  # biggest on top
    base = sens.base.p_fail
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.6))
    y = range(len(rows))
    lo_color, hi_color = "#2a6fdb", "#e07b39"
    for i, r in zip(y, rows):
        ax.barh(i, r.p_low - base, left=base, color=lo_color, height=0.6)
        ax.barh(i, r.p_high - base, left=base, color=hi_color, height=0.6)
    ax.axvline(base, color="#333", linewidth=1)
    ticks = [f"{r.knob.label}  ({r.knob.low_desc} / {r.knob.high_desc})" for r in rows]
    # Escape "$" so matplotlib does not treat dollar amounts as math text.
    ax.set_yticks(list(y), [t.replace("$", r"\$") for t in ticks], fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Probability of failure")
    ax.set_title(f"{sens.base.scenario.name}: what moves the failure rate "
                 f"(base {100 * base:.1f}%)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=lo_color, label="input lower"),
                       Patch(color=hi_color, label="input higher")],
              loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
