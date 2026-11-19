"""Turn simulated paths into decisions: summaries, comparisons, and charts."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import Result

PERCENTILES = (10, 50, 90)
Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% confidence interval for a proportion; well behaved near 0 and 1."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (lo, hi)


@dataclass
class Summary:
    name: str
    paths: int
    p_fail: float
    ci: tuple[float, float]
    p_end_below: float  # share of paths still below the floor at the horizon
    median_failure_year: float | None
    bands: dict[int, np.ndarray]  # percentile -> net worth by year (years + 1)
    terminal: dict[int, float]
    cumulative_failure: np.ndarray  # share of paths failed by end of each year


def summarize(res: Result) -> Summary:
    n = len(res.failed)
    k = int(res.failed.sum())
    bands = {q: np.percentile(res.net_worth, q, axis=0) for q in PERCENTILES}
    years = res.first_failure_year[res.failed]
    T = res.scenario.horizon_years
    counts = np.bincount(years, minlength=T)[:T] if k else np.zeros(T, dtype=int)
    return Summary(
        name=res.scenario.name,
        paths=n,
        p_fail=k / n,
        ci=wilson_interval(k, n),
        p_end_below=float((res.net_worth[:, -1] < res.scenario.failure_floor).mean()),
        median_failure_year=float(np.median(years)) if k else None,
        bands=bands,
        terminal={q: float(b[-1]) for q, b in bands.items()},
        cumulative_failure=np.cumsum(counts) / n,
    )


@dataclass
class Difference:
    name: str
    baseline: str
    delta: float  # change in failure probability (other - baseline)
    ci: tuple[float, float]
    delta_median_terminal: float


def compare(base: Result, other: Result) -> Difference:
    """Paired difference in failure probability under common random numbers.

    Because both scenarios saw the same random draws, each path is a matched
    pair, so the interval is much tighter than comparing two independent
    estimates.
    """
    if base.failed.shape != other.failed.shape:
        raise ValueError("scenarios must have the same number of paths to compare")
    d = other.failed.astype(float) - base.failed.astype(float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    delta = float(d.mean())
    return Difference(
        name=other.scenario.name,
        baseline=base.scenario.name,
        delta=delta,
        ci=(delta - Z95 * se, delta + Z95 * se),
        delta_median_terminal=float(
            np.median(other.net_worth[:, -1]) - np.median(base.net_worth[:, -1])
        ),
    )


@dataclass
class CareerMetrics:
    name: str
    age: int
    net_worth_at_age: dict[int, float]  # percentile -> portfolio at that age
    p_reach: float  # chance the portfolio is at or above target by that age
    lifetime_income: float  # median, all income before tax, through that age
    lifetime_tax: float  # median, through that age
    peak_income: float  # median of each path's best year
    years_in_debt: float  # mean years with a negative balance, through that age


def career_metrics(res: Result, age: int, target: float) -> CareerMetrics:
    """Career-oriented summary at a chosen age (or year, without start_age)."""
    s = res.scenario
    t = int(np.clip(age - (s.start_age or 0), 0, s.horizon_years))
    nw = res.net_worth[:, t]
    inc = res.income if res.income is not None else np.zeros_like(res.cash_flow)
    tax = res.tax if res.tax is not None else np.zeros_like(res.cash_flow)
    return CareerMetrics(
        name=s.name,
        age=age,
        net_worth_at_age={q: float(np.percentile(nw, q)) for q in PERCENTILES},
        p_reach=float((nw >= target).mean()),
        lifetime_income=float(np.median(inc[:, :t].sum(axis=1))),
        lifetime_tax=float(np.median(tax[:, :t].sum(axis=1))),
        peak_income=float(np.median(inc.max(axis=1))),
        years_in_debt=float((res.net_worth[:, 1 : t + 1] < 0).sum(axis=1).mean()),
    )


# ----------------------------------------------------------------------------
# Milestones
# ----------------------------------------------------------------------------

INCOME_MILESTONES = (200_000, 250_000, 300_000)  # take-home, today's dollars
NET_WORTH_MILESTONES = (1e6, 3e6, 5e6, 10e6, 20e6, 30e6)


@dataclass
class Milestone:
    level: float
    p_ever: float  # share of paths that reach it within the horizon
    median_age: float | None  # among paths that reach it
    early_age: float | None  # 10th percentile age among those paths (a fast path)
    by_age: np.ndarray  # share of paths that have reached it by each year
    gross_needed: float | None = None  # income milestones: gross pay for this take-home
    dividend: float | None = None  # net worth milestones: perpetual yearly draw


def _first_reach(series: np.ndarray, level: float) -> np.ndarray:
    hit = series >= level
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def _milestone(res: Result, series: np.ndarray, level: float) -> Milestone:
    a0 = res.scenario.start_age or 0
    first = _first_reach(series, level)
    reached = first >= 0
    ages = first[reached] + a0
    counts = np.bincount(first[reached], minlength=series.shape[1])[: series.shape[1]]
    return Milestone(
        level=level,
        p_ever=float(reached.mean()),
        median_age=float(np.median(ages)) if reached.any() else None,
        early_age=float(np.percentile(ages, 10)) if reached.any() else None,
        by_age=np.cumsum(counts) / len(first),
    )


def gross_for_take_home(s, take_home: float) -> float:
    """Gross taxable pay whose take-home equals ``take_home`` under the scenario's taxes."""
    from .engine import progressive_tax

    if not s.taxes:
        return take_home
    lo, hi = take_home, take_home * 3 + 100_000
    for _ in range(60):
        mid = (lo + hi) / 2
        net = mid - float(progressive_tax(np.array([mid]), s.taxes)[0])
        lo, hi = (mid, hi) if net < take_home else (lo, mid)
    return hi


def income_milestones(res: Result, levels=INCOME_MILESTONES) -> list[Milestone]:
    """When yearly take-home pay first reaches each level (today's dollars)."""
    th = res.take_home if res.take_home is not None else res.cash_flow
    out = []
    for level in levels:
        m = _milestone(res, th, level)
        m.gross_needed = gross_for_take_home(res.scenario, level)
        out.append(m)
    return out


def net_worth_milestones(
    res: Result, levels=NET_WORTH_MILESTONES, withdrawal_rate: float | None = None
) -> list[Milestone]:
    """When the portfolio first reaches each level, plus the perpetual draw it supports."""
    if withdrawal_rate is None:
        ls = res.scenario.lifestyle
        withdrawal_rate = ls.withdrawal_rate if ls else 0.035
    out = []
    for level in levels:
        m = _milestone(res, res.net_worth[:, 1:], level)
        m.dividend = level * withdrawal_rate
        out.append(m)
    return out


WEEKS_PER_YEAR = 48  # working weeks, net of vacation and holidays


@dataclass
class HoursMetrics:
    name: str
    lifetime_hours: float  # median over paths
    years_worked: float  # median count of years with any work hours
    by_decade: dict[str, float]  # mean hours/week while working, per age decade
    take_home_per_hour: float  # median lifetime take-home / lifetime hours
    lifetime_take_home: float  # median


def hours_metrics(res: Result) -> HoursMetrics | None:
    if res.hours is None:
        return None
    h = res.hours
    a0 = res.scenario.start_age or 0
    ages = np.arange(h.shape[1]) + a0
    total = (h * WEEKS_PER_YEAR).sum(axis=1)
    th = res.take_home.sum(axis=1) if res.take_home is not None else np.zeros(len(h))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_hour = np.where(total > 0, th / total, np.nan)
    decades = {}
    for lo in range(20, 80, 10):
        cols = (ages >= lo) & (ages < lo + 10)
        if cols.any():
            block = h[:, cols]
            working = block[block > 0]
            decades[f"{lo}s"] = float(working.mean()) if working.size else 0.0
    return HoursMetrics(
        name=res.scenario.name,
        lifetime_hours=float(np.median(total)),
        years_worked=float(np.median((h > 0).sum(axis=1))),
        by_decade=decades,
        take_home_per_hour=float(np.nanmedian(per_hour)) if np.isfinite(per_hour).any() else 0.0,
        lifetime_take_home=float(np.median(th)),
    )


@dataclass
class DebtMetrics:
    name: str
    p_borrow: float  # share of paths that ever owe student or other loans
    peak_median: float  # median peak balance among borrowers
    debt_free_age: float | None  # median age the last loan is paid off, among borrowers


def debt_metrics(res: Result) -> DebtMetrics | None:
    if res.debt is None or not res.debt.any():
        return None
    d = res.debt
    borrowed = d.max(axis=1) > 0
    peak = d.max(axis=1)[borrowed]
    owed = d > 0.5
    last = owed.shape[1] - 1 - np.argmax(owed[:, ::-1], axis=1)
    a0 = res.scenario.start_age or 0
    return DebtMetrics(
        name=res.scenario.name,
        p_borrow=float(borrowed.mean()),
        peak_median=float(np.median(peak)),
        debt_free_age=float(np.median(last[borrowed] + a0)) if borrowed.any() else None,
    )


def strategy_metrics(res: Result) -> dict | None:
    """When the finance strategy (FIRE / coast) kicked in."""
    st = res.scenario.strategy
    if st is None or res.strategy_year is None:
        return None
    y = res.strategy_year
    hit = y >= 0
    a0 = res.scenario.start_age or 0
    return {
        "strategy": st.name,
        "p_hit": float(hit.mean()),
        "median_age": float(np.median(y[hit]) + a0) if hit.any() else None,
        "early_age": float(np.percentile(y[hit], 10) + a0) if hit.any() else None,
        "by_age": np.cumsum(np.bincount(y[hit], minlength=res.scenario.horizon_years)
                            [: res.scenario.horizon_years]) / len(y),
    }


def tier_shares(res: Result) -> dict[str, np.ndarray]:
    """Share of paths in each lifestyle tier, by year."""
    ls = res.scenario.lifestyle
    if res.tier is None or ls is None:
        return {}
    names = ["Base budget"] + [t.name for t in ls.tiers]
    return {n: (res.tier == k).mean(axis=0) for k, n in enumerate(names)}


def format_milestones(results: list[Result]) -> str:
    def cell(m: Milestone) -> str:
        if not m.p_ever:
            return "never"
        return f"{100 * m.p_ever:.0f}% @{m.median_age:.0f}"

    lines = []
    inc = [income_milestones(r) for r in results]
    if inc:
        head = "".join(f"{_money(m.level) + ' (' + _money(m.gross_needed) + ' gross)':>26}"
                       for m in inc[0])
        lines += ["", "Take-home pay milestones (today's $): chance ever reached @ median age",
                  f"{'':<22}{head}"]
        for r, ms in zip(results, inc):
            lines.append(f"{r.scenario.name[:21]:<22}" + "".join(f"{cell(m):>26}" for m in ms))
    nw = [net_worth_milestones(r) for r in results]
    if nw:
        head = "".join(f"{_money(m.level) + ' (' + _money(m.dividend) + '/yr)':>18}" for m in nw[0])
        lines += ["", "Net worth milestones: chance ever reached @ median age "
                      "(perpetual draw in brackets)", f"{'':<22}{head}"]
        for r, ms in zip(results, nw):
            lines.append(f"{r.scenario.name[:21]:<22}" + "".join(f"{cell(m):>18}" for m in ms))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Text output
# ----------------------------------------------------------------------------


def _money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.0f}k"
    return f"{sign}${x:.0f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _when(res: Result, year: float | None) -> str:
    if year is None:
        return "-"
    age = res.scenario.start_age
    return f"age {age + year:.0f}" if age is not None else f"year {year:.0f}"


def format_report(results: list[Result]) -> str:
    sums = [summarize(r) for r in results]
    s0 = results[0].scenario
    floor = f"below {_money(s0.failure_floor)}" if s0.failure_floor else "below zero"
    lines = [
        f"STATS Projections: {s0.paths:,} paths x {s0.horizon_years} years, seed {s0.seed}",
        f"P(fail) = portfolio drops {floor} at any point (even briefly).",
        f"Ends below = still {floor} at the end of the horizon. All dollars are today's dollars.",
        "",
    ]
    header = (
        f"{'Scenario':<22}{'P(fail)':>9}{'95% CI':>17}{'Ends below':>12}{'Median fail':>13}"
        f"{'P10 end':>11}{'P50 end':>11}{'P90 end':>11}"
    )
    lines += [header, "-" * len(header)]
    for res, s in zip(results, sums):
        ci = f"{_pct(s.ci[0])} - {_pct(s.ci[1])}"
        lines.append(
            f"{s.name[:21]:<22}{_pct(s.p_fail):>9}{ci:>17}{_pct(s.p_end_below):>12}"
            f"{_when(res, s.median_failure_year):>13}"
            f"{_money(s.terminal[10]):>11}{_money(s.terminal[50]):>11}{_money(s.terminal[90]):>11}"
        )
    if len(results) > 1:
        lines += ["", f"Versus {results[0].scenario.name} (same random draws, paired 95% CI):"]
        for r in results[1:]:
            d = compare(results[0], r)
            lines.append(
                f"  {d.name[:30]:<32}failure {100 * d.delta:+.1f} pts "
                f"[{100 * d.ci[0]:+.1f}, {100 * d.ci[1]:+.1f}]   "
                f"median end {'+' if d.delta_median_terminal >= 0 else ''}"
                f"{_money(d.delta_median_terminal)}"
            )
    lines += [
        "",
        "A failure rate describes your assumptions, not the future. Check how it moves",
        "when you change the inputs you are least sure of.",
    ]
    return "\n".join(lines)


def export_csv(results: list[Result], folder: str | Path) -> list[Path]:
    """One CSV per scenario: year, age, P10/P50/P90 net worth, cumulative failure."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for res in results:
        s = summarize(res)
        path = folder / f"{_slug(s.name)}.csv"
        age0 = res.scenario.start_age
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["year", "age", "p10", "p50", "p90", "cumulative_failure"])
            for t in range(res.scenario.horizon_years + 1):
                cum = 0.0 if t == 0 else s.cumulative_failure[t - 1]
                w.writerow([
                    t, "" if age0 is None else age0 + t,
                    round(s.bands[10][t], 2), round(s.bands[50][t], 2),
                    round(s.bands[90][t], 2), round(float(cum), 5),
                ])
        written.append(path)
    return written


def _slug(name: str) -> str:
    import re

    s = re.sub(r"[^a-z0-9]+", "_", name.lower().replace("&", "and"))
    return s.strip("_") or "scenario"


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------

COLORS = ("#2a6fdb", "#e07b39", "#2e9e6a", "#a94fc9", "#c9474f", "#6b7a8f")


def plot(results: list[Result], path: str | Path) -> Path:
    """Percentile bands and cumulative failure probability, one color per scenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}
    )
    for i, res in enumerate(results):
        s = summarize(res)
        color = COLORS[i % len(COLORS)]
        age0 = res.scenario.start_age or 0
        x = np.arange(res.scenario.horizon_years + 1) + age0
        ax1.fill_between(x, s.bands[10], s.bands[90], color=color, alpha=0.15, linewidth=0)
        ax1.plot(x, s.bands[50], color=color, linewidth=2,
                 label=f"{s.name}  (fail {_pct(s.p_fail)})")
        ax1.plot(x, s.bands[10], color=color, linewidth=0.8, linestyle="--")
        ax2.plot(x[1:], s.cumulative_failure, color=color, linewidth=2)

    floor = results[0].scenario.failure_floor
    ax1.axhline(floor, color="#555", linewidth=0.8)
    ax1.set_ylabel("Portfolio (today's $)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(v)))
    ax1.set_title("Median with P10-P90 band (dashed = P10)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.25)
    ax2.set_ylabel("Failed by now")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Age" if results[0].scenario.start_age is not None else "Year")
    ax2.grid(alpha=0.25)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Failure types, startup runway, free time
# ----------------------------------------------------------------------------

FAILURE_MODES = {
    "crunch": "Cash crunch: liquid assets below the floor at any point",
    "insolvent": "Negative net worth (loans exceed assets) at 40 or later",
    "retirement": "Runs out in retirement: liquid assets below the floor after 65",
}


def failure_modes(res: Result, insolvency_age: int = 40, retire_age: int = 65) -> dict:
    """Per path flags and first-failure years for three different failures.

    * crunch: the portfolio (cash + investments) drops below the floor, even
      briefly, e.g. a job gap or a down payment. This is ``res.failed``.
    * insolvent: net worth (including home equity and loans) is below 0 at or
      after ``insolvency_age``, after school and training should be over.
    * retirement: the portfolio drops below the floor at or after
      ``retire_age``: the classic "money ran out" outcome.
    """
    s = res.scenario
    a0 = s.start_age or 0
    ages = a0 + 1 + np.arange(s.horizon_years)
    liquid = res.portfolio[:, 1:] if res.portfolio is not None else res.net_worth[:, 1:]
    below = liquid < s.failure_floor
    masks = {
        "crunch": below,
        "insolvent": (res.net_worth[:, 1:] < 0) & (ages >= insolvency_age),
        "retirement": below & (ages >= retire_age),
    }
    out = {}
    for k, m in masks.items():
        ever = m.any(axis=1)
        out[k] = {"p": float(ever.mean()), "cumulative": np.cumsum(m, axis=1).astype(bool).mean(axis=0)}
    return out


def runway_metrics(res: Result) -> dict | None:
    """How a startup runway held up: None unless the career has a runway window."""
    s = res.scenario
    if s.cash is None or not s.cash.runway_window or res.cash is None:
        return None
    a, b = s.cash.runway_window
    b = min(b, s.horizon_years)
    if b <= a:
        return None
    cash = res.cash[:, a + 1:b + 1]
    liquid = res.portfolio[:, a + 1:b + 1]
    return {
        "window": (a, b),
        "cash_at_start": float(np.median(res.cash[:, a])),
        "peak_cash": float(np.median(res.cash[:, a:b + 1].max(axis=1))),
        "p_cash_out": float((cash < 1.0).any(axis=1).mean()),
        "p_broke": float((liquid < s.failure_floor).any(axis=1).mean()),
    }


AWAKE_HOURS_PER_WEEK = 16 * 7
COMMUTE_HOURS_PER_WEEK = 4.5  # Census ACS: ~27 minutes each way, 5 days a week


@dataclass(frozen=True)
class FreeTime:
    name: str
    lifetime_free: float  # waking hours not spent working or commuting, now to the plan's end
    free_before_65: float
    working_years: float
    retired_years: float
    free_share: float  # free hours / waking hours over the whole plan


def free_time(res: Result, commute: float = COMMUTE_HOURS_PER_WEEK,
              retire_age: int = 65) -> FreeTime | None:
    """Waking hours left after work and commuting (vacation weeks count as free)."""
    if res.hours is None:
        return None
    s = res.scenario
    a0 = s.start_age or 0
    h = res.hours
    working = h > 0
    busy = (h + np.where(working, commute, 0.0)) * WEEKS_PER_YEAR
    free = AWAKE_HOURS_PER_WEEK * 52 - busy
    ages = a0 + np.arange(s.horizon_years)
    return FreeTime(
        name=s.name,
        lifetime_free=float(np.median(free.sum(axis=1))),
        free_before_65=float(np.median(free[:, ages < retire_age].sum(axis=1))),
        working_years=float(np.median(working.sum(axis=1))),
        retired_years=float(np.median((~working & (ages >= a0 + 4)).sum(axis=1))),
        free_share=float(np.median(free.sum(axis=1)) / (AWAKE_HOURS_PER_WEEK * 52 * s.horizon_years)),
    )
