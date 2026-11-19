"""Scenario files: loading, inheritance, and validation.

A scenario is a YAML file describing one possible future. A scenario can
``extends:`` another file and override only what differs. Merge rules:

* mappings merge key by key, recursively;
* lists of named items (income, expenses, events) merge by ``name``: a
  matching item has its fields overridden, a new name is appended, and an
  item with ``remove: true`` is dropped;
* any other value is replaced outright.

All money is in today's (real, inflation adjusted) dollars. Year 0 is the
first simulated year. A stream is active for ``start_year <= year < end_year``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


class ScenarioError(ValueError):
    """Raised when a scenario file is missing, malformed, or inconsistent."""


# ----------------------------------------------------------------------------
# Typed model
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Growth:
    """Yearly real growth rate drawn as mean + stdev * N(0, 1)."""

    mean: float = 0.0
    stdev: float = 0.0


@dataclass(frozen=True)
class Stream:
    """A recurring yearly income or expense."""

    name: str
    amount: float
    start_year: int = 0
    end_year: int | None = None
    growth: Growth = Growth()
    loss_probability: float = 0.0  # chance per year the stream is cut (job loss)
    loss_fraction: float = 1.0  # share of that year's amount lost when it is cut
    noise: float = 0.0  # year-to-year swing that does not compound (bonus, commission)
    taxable: bool = False  # True: amount is gross pay, taxed with the scenario's taxes
    spendable: bool = True  # False: goes straight to savings (employer 401(k) match)


@dataclass(frozen=True)
class Tier:
    """A quality-of-life tier: once affordable income reaches ``take_home``,
    the listed budget categories switch to these monthly amounts."""

    name: str
    take_home: float
    monthly: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Lifestyle:
    """How spending grows with income.

    Each year, affordable income = average take-home over the last
    ``smoothing_years`` + ``withdrawal_rate`` x portfolio (a perpetual draw).
    The highest tier at or below that income sets the listed categories, and
    ``creep`` of every dollar above that tier's threshold is spent too.
    """

    tiers: tuple[Tier, ...] = ()
    creep: float = 0.0
    withdrawal_rate: float = 0.035
    smoothing_years: int = 3
    # Tier budgets are multiplied by the index in force each year:
    # ((year, index), ...), e.g. ((2, 0.6),) after moving somewhere cheaper.
    cost_index: tuple[tuple[int, float], ...] = ()
    max_tier: int | None = None  # cap the tier you allow yourself (lean living)

    def index_by_year(self, years: int):
        import numpy as np

        out = np.ones(years)
        for y, idx in sorted(self.cost_index):
            out[max(y, 0):] = idx
        return out

    @property
    def categories(self) -> tuple[str, ...]:
        seen: list[str] = []
        for t in self.tiers:
            for c, _ in t.monthly:
                if c not in seen:
                    seen.append(c)
        return tuple(seen)


EVENT_TYPES = ("purchase", "windfall", "set_amount", "loan", "scale")


@dataclass(frozen=True)
class TaxLayer:
    """One progressive tax: brackets are (threshold, marginal rate) pairs.

    ``per_earner``: applied to each taxable income stream separately (payroll
    taxes such as Social Security, whose wage cap is per person), instead of
    to the household's combined taxable income.
    """

    name: str
    brackets: tuple[tuple[float, float], ...]
    deduction: float = 0.0
    per_earner: bool = False


STRATEGY_KINDS = ("work", "fire", "coast")


@dataclass(frozen=True)
class Strategy:
    """A rule for when to stop (or ease off) working.

    * ``work``: work until the scheduled retirement events.
    * ``fire``: stop the ``stop`` streams for good once the portfolio is at
      least ``multiple`` x that year's spending (25x = the 4% rule), from
      ``min_year`` on.
    * ``coast``: once the portfolio would grow to ``multiple`` x spending by
      ``target_year`` with no more saving (at ``expected_real_return``),
      scale the ``stop`` streams by ``coast_factor`` (an easier, part-time or
      lower-paid job) until the normal retirement events.
    """

    name: str = "Work to 65"
    kind: str = "work"
    multiple: float = 25.0
    min_year: int = 0
    stop: tuple[str, ...] = ()
    coast_factor: float = 0.5
    expected_real_return: float = 0.045
    target_year: int = 36


@dataclass(frozen=True)
class Event:
    """A one off cash flow or a step change in a stream.

    * ``purchase``: one time outflow of ``amount``.
    * ``windfall``: one time inflow of ``amount``.
    * ``set_amount``: from this year on, stream ``target`` is ``amount``
      (its growth continues from the new level). Use 0 to end a stream,
      for example at retirement.

      With ``duration``, the change lasts that many years and then the
      stream returns to where it would have been (school, sabbatical).
    * ``loan``: receive ``amount`` now; after ``defer`` years (interest
      accrues meanwhile) repay it in equal real payments over ``term`` years
      at nominal ``rate``.

    Timing uncertainty: ``jitter`` moves the year uniformly by up to that many
    years in either direction, independently per path. ``same_time_as`` ties
    an event to another event's (jittered) year plus ``offset``, so a down
    payment and the new housing cost always land together.

    Outcome uncertainty: ``probability`` is the chance the event happens at
    all. Events sharing a ``group`` are mutually exclusive outcomes (their
    probabilities must sum to at most 1). An event linked with
    ``same_time_as`` only happens on paths where its anchor happened.
    ``spread`` makes the amount lognormal with median ``amount``.
    ``tax_rate`` is a flat tax on a windfall.
    """

    name: str
    type: str
    year: int = 0
    amount: float = 0.0
    target: str | None = None
    jitter: int = 0
    same_time_as: str | None = None
    offset: int = 0
    probability: float = 1.0
    group: str | None = None
    spread: float = 0.0
    tax_rate: float = 0.0
    duration: int | None = None
    rate: float = 0.0
    term: int = 10
    defer: int = 0
    factor: float = 1.0  # scale events: multiply the target stream from this year on


@dataclass(frozen=True)
class Returns:
    """Nominal yearly portfolio return model."""

    model: str = "normal"  # normal | student_t | bootstrap
    mean: float = 0.07
    stdev: float = 0.15
    dof: float = 5.0  # student_t only
    data: str | None = None  # bootstrap only: CSV path with year,return,inflation
    block_size: int = 1  # bootstrap only: consecutive years sampled together


@dataclass(frozen=True)
class Inflation:
    mean: float = 0.025
    stdev: float = 0.01
    correlation: float = 0.0  # with nominal returns (normal / student_t models)


@dataclass(frozen=True)
class Scenario:
    name: str
    horizon_years: int
    paths: int
    seed: int
    starting_balance: float
    returns: Returns
    inflation: Inflation
    income: tuple[Stream, ...]
    expenses: tuple[Stream, ...]
    events: tuple[Event, ...] = ()
    failure_floor: float = 0.0
    borrowing_rate: float = 0.05  # real rate charged while the balance is negative
    start_age: int | None = None
    taxes: tuple[TaxLayer, ...] = ()
    lifestyle: Lifestyle | None = None
    time: tuple[Stream, ...] = ()  # work hours per week (amount = hours, not dollars)
    strategy: Strategy | None = None
    source: Path | None = field(default=None, compare=False)

    def stream(self, name: str) -> Stream:
        for s in (*self.income, *self.expenses, *self.time):
            if s.name == name:
                return s
        raise KeyError(name)


# ----------------------------------------------------------------------------
# Loading and merging
# ----------------------------------------------------------------------------


NAMED_LISTS = ("income", "expenses", "events", "time", "taxes")


def _merge(base: Any, override: Any, key: str | None = None) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _merge(base[k], v, k) if k in base else v
        return out
    if key in NAMED_LISTS and isinstance(base, list) and isinstance(override, list):
        out = [dict(item) for item in base]
        index = {item.get("name"): i for i, item in enumerate(out)}
        for item in override:
            if not isinstance(item, dict) or "name" not in item:
                raise ScenarioError(f"every item in '{key}' needs a name: {item!r}")
            name = item["name"]
            if name in index:
                out[index[name]] = _merge(out[index[name]], item)
            else:
                index[name] = len(out)
                out.append(dict(item))
        return [item for item in out if not item.get("remove", False)]
    if key in ("sources", "assumptions") and isinstance(base, list) and isinstance(override, list):
        return base + [s for s in override if s not in base]
    return override


def load_raw(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Read a scenario file and resolve ``extends`` and ``include`` into one dict.

    Order: the ``extends`` parent, then each ``include`` in turn (for shared
    pieces such as a tax table or a career path), then this file's own keys.
    """
    path = Path(path).resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ScenarioError(f"circular 'extends': {chain}")
    if not path.exists():
        raise ScenarioError(f"scenario file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ScenarioError(f"{path.name}: top level must be a mapping")

    parent = data.pop("extends", None)
    includes = data.pop("include", None) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict = {}
    if parent is not None:
        merged = load_raw(path.parent / parent, (*_seen, path))
        merged.pop("name", None)  # a child never inherits its parent's name
    for inc in includes:
        part = load_raw(path.parent / inc, (*_seen, path))
        for k in ("name", "description"):
            part.pop(k, None)
        merged = _merge(merged, part)
    merged = _merge(merged, data)

    # Relative data paths resolve against the file that declared them.
    returns = data.get("portfolio", {}).get("returns", {})
    if isinstance(returns, dict) and returns.get("data"):
        merged["portfolio"]["returns"]["data"] = str(
            (path.parent / returns["data"]).resolve()
        )
    return merged


def load_scenario(path: str | Path, **overrides: Any) -> Scenario:
    """Load, merge, and validate a scenario. ``overrides`` replace top level keys."""
    raw = load_raw(path)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    raw.setdefault("name", Path(path).stem)
    return replace(from_dict(raw), source=Path(path).resolve())


def compose(profile: str | Path, career: str | Path, *extras: str | Path,
            **overrides: Any) -> Scenario:
    """A personal profile (who you are today), a career path, and optional
    add-ons layered on top in order: life events (partner, kids, moving
    abroad) and a finance strategy (FIRE, coast FIRE).

    Each file merges by name, as if it had ``extends:`` the one before it.
    The career's name stays the scenario's name.
    """
    base = load_raw(profile)
    base.pop("name", None)
    raw = _merge(base, load_raw(career))
    for extra in extras:
        part = load_raw(extra)
        for k in ("name", "description", "sources", "assumptions"):
            part.pop(k, None)
        raw = _merge(raw, part)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    raw.setdefault("name", Path(career).stem)
    return replace(from_dict(raw), source=Path(career).resolve())


def career_info(path: str | Path) -> dict:
    """Name, description, and sources of a career file, without validating it."""
    raw = load_raw(path)
    return {"name": raw.get("name", Path(path).stem),
            "description": raw.get("description", ""),
            "sources": raw.get("sources", []),
            "assumptions": raw.get("assumptions", [])}


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


def _check_keys(where: str, data: dict, allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ScenarioError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def _growth(where: str, data: Any) -> Growth:
    if data is None:
        return Growth()
    if isinstance(data, (int, float)):
        return Growth(mean=float(data))
    _check_keys(where, data, {"mean", "stdev"})
    g = Growth(float(data.get("mean", 0.0)), float(data.get("stdev", 0.0)))
    if g.stdev < 0:
        raise ScenarioError(f"{where}: stdev must be >= 0")
    return g


def _stream(kind: str, data: dict, horizon: int) -> Stream:
    where = f"{kind} '{data.get('name', '?')}'"
    _check_keys(
        where,
        data,
        {"name", "amount", "start_year", "end_year", "growth",
         "loss_probability", "loss_fraction", "noise", "taxable", "spendable", "monthly",
         "remove", "note"},
    )
    if ("amount" in data) == ("monthly" in data):
        raise ScenarioError(f"{where}: give exactly one of 'amount' (per year) or 'monthly'")
    amount = float(data["amount"]) if "amount" in data else 12 * float(data["monthly"])
    s = Stream(
        name=str(data["name"]),
        amount=amount,
        start_year=int(data.get("start_year", 0)),
        end_year=None if data.get("end_year") is None else int(data["end_year"]),
        growth=_growth(f"{where} growth", data.get("growth")),
        loss_probability=float(data.get("loss_probability", 0.0)),
        loss_fraction=float(data.get("loss_fraction", 1.0)),
        noise=float(data.get("noise", 0.0)),
        taxable=bool(data.get("taxable", False)),
        spendable=bool(data.get("spendable", True)),
    )
    if s.amount < 0:
        raise ScenarioError(f"{where}: amount must be >= 0 (use expenses for outflows)")
    if s.noise < 0:
        raise ScenarioError(f"{where}: noise must be >= 0")
    if s.taxable and kind != "income":
        raise ScenarioError(f"{where}: only income can be taxable")
    if s.end_year is not None and s.end_year < s.start_year:
        raise ScenarioError(f"{where}: end_year is before start_year")
    if not 0 <= s.loss_probability <= 1 or not 0 <= s.loss_fraction <= 1:
        raise ScenarioError(f"{where}: loss_probability and loss_fraction must be in [0, 1]")
    return s


def _event(data: dict) -> Event:
    where = f"event '{data.get('name', '?')}'"
    _check_keys(
        where,
        data,
        {"name", "type", "year", "amount", "target", "jitter", "same_time_as", "offset",
         "probability", "group", "spread", "tax_rate", "duration", "rate", "term", "defer",
         "factor", "remove", "note"},
    )
    e = Event(
        name=str(data["name"]),
        type=str(data.get("type", "")),
        year=int(data.get("year", 0)),
        amount=float(data.get("amount", 0.0)),
        target=data.get("target"),
        jitter=int(data.get("jitter", 0)),
        same_time_as=data.get("same_time_as"),
        offset=int(data.get("offset", 0)),
        probability=float(data.get("probability", 1.0)),
        group=data.get("group"),
        spread=float(data.get("spread", 0.0)),
        tax_rate=float(data.get("tax_rate", 0.0)),
        duration=None if data.get("duration") is None else int(data["duration"]),
        rate=float(data.get("rate", 0.0)),
        term=int(data.get("term", 10)),
        defer=int(data.get("defer", 0)),
        factor=float(data.get("factor", 1.0)),
    )
    if e.factor < 0:
        raise ScenarioError(f"{where}: factor must be >= 0")
    if e.type not in EVENT_TYPES:
        raise ScenarioError(f"{where}: type must be one of {EVENT_TYPES}")
    if e.amount < 0:
        raise ScenarioError(f"{where}: amount must be >= 0")
    if e.jitter < 0:
        raise ScenarioError(f"{where}: jitter must be >= 0")
    if not 0 <= e.probability <= 1 or not 0 <= e.tax_rate < 1:
        raise ScenarioError(f"{where}: probability must be in [0, 1] and tax_rate in [0, 1)")
    if e.spread < 0:
        raise ScenarioError(f"{where}: spread must be >= 0")
    if e.duration is not None and (e.duration < 1 or e.type not in ("set_amount", "scale")):
        raise ScenarioError(f"{where}: duration must be >= 1 and only applies to set_amount or scale")
    if e.type == "scale" and not e.target:
        raise ScenarioError(f"{where}: scale needs a 'target' stream")
    if e.type == "loan" and (e.term < 1 or e.defer < 0 or e.rate < 0):
        raise ScenarioError(f"{where}: loan needs term >= 1, defer >= 0, rate >= 0")
    if e.type == "set_amount" and not e.target:
        raise ScenarioError(f"{where}: set_amount needs a 'target' stream")
    if "year" not in data and not e.same_time_as:
        raise ScenarioError(f"{where}: needs 'year' or 'same_time_as'")
    return e


TOP_KEYS = {
    "name", "description", "horizon_years", "paths", "seed", "start_age",
    "failure_floor", "borrowing_rate", "portfolio", "income", "expenses", "events",
    "taxes", "sources", "assumptions", "lifestyle", "time", "strategy",
}


def _strategy(data: Any) -> Strategy | None:
    if not data:
        return None
    _check_keys("strategy", data, set(Strategy.__dataclass_fields__) | {"note"})
    d = {k: v for k, v in data.items() if k != "note"}
    if "stop" in d:
        d["stop"] = tuple(d["stop"] or ())
    st = Strategy(**d)
    if st.kind not in STRATEGY_KINDS:
        raise ScenarioError(f"strategy.kind must be one of {STRATEGY_KINDS}")
    if st.multiple <= 0 or not 0 <= st.coast_factor <= 1:
        raise ScenarioError("strategy: multiple > 0 and coast_factor in [0, 1]")
    return st


def _expand_targets(events: list[dict]) -> list[dict]:
    """``target: [A, B, C]`` becomes one event per stream, all happening together.

    The first keeps the event's name, timing and odds; the rest are linked to
    it with ``same_time_as`` so they share its year and only happen with it.
    """
    out = []
    for d in events:
        targets = d.get("target")
        if not isinstance(targets, list):
            out.append(d)
            continue
        if not targets:
            raise ScenarioError(f"event '{d.get('name', '?')}': empty target list")
        out.append({**d, "target": targets[0]})
        for t in targets[1:]:
            linked = {k: v for k, v in d.items()
                      if k not in ("year", "jitter", "probability", "group", "same_time_as",
                                   "offset", "target")}
            linked.update(name=f"{d['name']} ({t})", target=t, same_time_as=d["name"])
            out.append(linked)
    return out


def _lifestyle(data: Any, expense_names: list[str]) -> Lifestyle | None:
    if not data:
        return None
    _check_keys("lifestyle", data, {"tiers", "creep", "withdrawal_rate", "smoothing_years",
                                    "cost_index", "max_tier", "note"})
    tiers = []
    for d in data.get("tiers", []) or []:
        where = f"lifestyle tier '{d.get('name', '?')}'"
        _check_keys(where, d, {"name", "take_home", "monthly", "note"})
        monthly = tuple((str(k), float(v)) for k, v in (d.get("monthly") or {}).items())
        unknown = [k for k, _ in monthly if k not in expense_names]
        if unknown:
            raise ScenarioError(f"{where}: unknown expense categories {unknown}")
        if any(v < 0 for _, v in monthly):
            raise ScenarioError(f"{where}: monthly amounts must be >= 0")
        tiers.append(Tier(str(d["name"]), float(d["take_home"]), monthly))
    if [t.take_home for t in tiers] != sorted(t.take_home for t in tiers):
        raise ScenarioError("lifestyle tiers must be in increasing take_home order")
    ls = Lifestyle(
        tiers=tuple(tiers),
        creep=float(data.get("creep", 0.0)),
        withdrawal_rate=float(data.get("withdrawal_rate", 0.035)),
        smoothing_years=int(data.get("smoothing_years", 3)),
        cost_index=tuple((int(y), float(i)) for y, i in (data.get("cost_index") or ())),
        max_tier=None if data.get("max_tier") is None else int(data["max_tier"]),
    )
    if any(i <= 0 for _, i in ls.cost_index):
        raise ScenarioError("lifestyle.cost_index values must be > 0")
    if not 0 <= ls.creep <= 1 or not 0 <= ls.withdrawal_rate < 0.2 or ls.smoothing_years < 1:
        raise ScenarioError("lifestyle: creep in [0, 1], withdrawal_rate in [0, 0.2), smoothing_years >= 1")
    return ls


def _taxes(data: Any) -> tuple[TaxLayer, ...]:
    if not data:
        return ()
    layers = data.get("layers", data) if isinstance(data, dict) else data
    out = []
    for d in layers:
        where = f"tax '{d.get('name', '?')}'"
        _check_keys(where, d, {"name", "brackets", "deduction", "per_earner", "note"})
        br = tuple((float(a), float(b)) for a, b in d["brackets"])
        if [a for a, _ in br] != sorted(a for a, _ in br) or br[0][0] != 0:
            raise ScenarioError(f"{where}: brackets must start at 0 and increase")
        if any(not 0 <= r < 1 for _, r in br):
            raise ScenarioError(f"{where}: rates must be decimals in [0, 1)")
        out.append(TaxLayer(str(d["name"]), br, float(d.get("deduction", 0.0)),
                            bool(d.get("per_earner", False))))
    return tuple(out)


def from_dict(raw: dict) -> Scenario:
    name = str(raw.get("name", "scenario"))
    _check_keys(f"scenario '{name}'", raw, TOP_KEYS)

    horizon = int(raw.get("horizon_years", 50))
    paths = int(raw.get("paths", 10_000))
    if horizon < 1 or paths < 1:
        raise ScenarioError("horizon_years and paths must be positive")

    portfolio = raw.get("portfolio", {}) or {}
    _check_keys("portfolio", portfolio, {"starting_balance", "returns", "inflation"})
    r = portfolio.get("returns", {}) or {}
    _check_keys("portfolio.returns", r, set(Returns.__dataclass_fields__))
    returns = Returns(**r)
    if returns.model not in ("normal", "student_t", "bootstrap"):
        raise ScenarioError("portfolio.returns.model must be normal, student_t, or bootstrap")
    if returns.model == "student_t" and returns.dof <= 2:
        raise ScenarioError("portfolio.returns.dof must be > 2 so the variance exists")
    if returns.model == "bootstrap" and not returns.data:
        raise ScenarioError("bootstrap returns need 'data': a CSV of year,return,inflation")
    if returns.block_size < 1:
        raise ScenarioError("portfolio.returns.block_size must be >= 1")
    i = portfolio.get("inflation", {}) or {}
    _check_keys("portfolio.inflation", i, set(Inflation.__dataclass_fields__))
    inflation = Inflation(**i)
    if not -1 <= inflation.correlation <= 1:
        raise ScenarioError("portfolio.inflation.correlation must be in [-1, 1]")

    income = tuple(_stream("income", d, horizon) for d in raw.get("income", []) or [])
    expenses = tuple(_stream("expense", d, horizon) for d in raw.get("expenses", []) or [])
    time = tuple(_stream("time", d, horizon) for d in raw.get("time", []) or [])
    events = tuple(_event(d) for d in _expand_targets(list(raw.get("events", []) or [])))

    names = [s.name for s in (*income, *expenses, *time)]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ScenarioError(f"stream names must be unique across income and expenses: {sorted(dupes)}")
    event_names = [e.name for e in events]
    if len(set(event_names)) != len(event_names):
        raise ScenarioError("event names must be unique")
    for e in events:
        if e.target and e.target not in names:
            raise ScenarioError(f"event '{e.name}': unknown target stream '{e.target}'")
        if e.same_time_as and e.same_time_as not in event_names:
            raise ScenarioError(f"event '{e.name}': unknown same_time_as event '{e.same_time_as}'")
    groups: dict[str, float] = {}
    for e in events:
        if e.group:
            groups[e.group] = groups.get(e.group, 0.0) + e.probability
    over = {g: p for g, p in groups.items() if p > 1 + 1e-9}
    if over:
        raise ScenarioError(f"outcome group probabilities sum above 1: {over}")
    taxes = _taxes(raw.get("taxes"))
    if any(s.taxable for s in income) and not taxes:
        raise ScenarioError("some income is taxable but the scenario has no 'taxes' section")

    return Scenario(
        name=name,
        horizon_years=horizon,
        paths=paths,
        seed=int(raw.get("seed", 42)),
        starting_balance=float(portfolio.get("starting_balance", 0.0)),
        returns=returns,
        inflation=inflation,
        income=income,
        expenses=expenses,
        events=events,
        failure_floor=float(raw.get("failure_floor", 0.0)),
        borrowing_rate=float(raw.get("borrowing_rate", 0.05)),
        start_age=None if raw.get("start_age") is None else int(raw["start_age"]),
        taxes=taxes,
        lifestyle=_lifestyle(raw.get("lifestyle"), [s.name for s in expenses]),
        time=time,
        strategy=_strategy(raw.get("strategy")),
    )
