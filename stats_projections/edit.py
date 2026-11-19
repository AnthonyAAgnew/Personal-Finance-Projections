"""Turn a Scenario back into scenario-file data, and diff two of them.

Used by the Streamlit app: the user edits a copy of a scenario, and the
edits are saved as a small override file that ``extends`` the original, so
the "configuration over code" files stay the source of truth.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import Event, Scenario, Stream

_STREAM_DEFAULTS = {"start_year": 0, "end_year": None, "loss_probability": 0.0,
                    "loss_fraction": 1.0, "noise": 0.0, "taxable": False, "spendable": True}
_EVENT_DEFAULTS = {"year": 0, "amount": 0.0, "target": None, "jitter": 0,
                   "same_time_as": None, "offset": 0, "probability": 1.0, "group": None,
                   "spread": 0.0, "tax_rate": 0.0, "duration": None, "rate": 0.0,
                   "term": 10, "defer": 0, "factor": 1.0}


def _clean(d: dict, defaults: dict) -> dict:
    return {k: v for k, v in d.items() if k not in defaults or v != defaults[k]}


def stream_to_dict(s: Stream) -> dict:
    d = asdict(s)
    if d["growth"] == {"mean": 0.0, "stdev": 0.0}:
        del d["growth"]
    return _clean(d, _STREAM_DEFAULTS)


def event_to_dict(e: Event) -> dict:
    d = _clean(asdict(e), _EVENT_DEFAULTS)
    if e.same_time_as:
        d.pop("year", None)  # the year comes from the linked event
    elif "year" not in d:
        d["year"] = 0  # 'year' is required unless same_time_as is set
    return d


def to_dict(s: Scenario) -> dict:
    returns = {k: v for k, v in asdict(s.returns).items() if v is not None}
    if s.returns.model != "student_t":
        returns.pop("dof", None)
    if s.returns.model != "bootstrap":
        returns.pop("data", None)
        returns.pop("block_size", None)
    out: dict[str, Any] = {
        "name": s.name,
        "start_age": s.start_age,
        "horizon_years": s.horizon_years,
        "paths": s.paths,
        "seed": s.seed,
        "failure_floor": s.failure_floor,
        "borrowing_rate": s.borrowing_rate,
        "portfolio": {
            "starting_balance": s.starting_balance,
            "returns": returns,
            "inflation": asdict(s.inflation),
        },
        "income": [stream_to_dict(x) for x in s.income],
        "expenses": [stream_to_dict(x) for x in s.expenses],
        "events": [event_to_dict(e) for e in s.events],
    }
    if out["start_age"] is None:
        del out["start_age"]
    if s.taxes:
        out["taxes"] = [
            {"name": t.name, "brackets": [list(b) for b in t.brackets],
             **({"deduction": t.deduction} if t.deduction else {}),
             **({"per_earner": True} if t.per_earner else {})}
            for t in s.taxes
        ]
    if s.lifestyle is not None:
        ls = s.lifestyle
        out["lifestyle"] = {
            "creep": ls.creep,
            "withdrawal_rate": ls.withdrawal_rate,
            "smoothing_years": ls.smoothing_years,
            "tiers": [{"name": t.name, "take_home": t.take_home, "monthly": dict(t.monthly)}
                      for t in ls.tiers],
        }
        if ls.cost_index:
            out["lifestyle"]["cost_index"] = [list(x) for x in ls.cost_index]
        if ls.max_tier is not None:
            out["lifestyle"]["max_tier"] = ls.max_tier
    if s.time:
        out["time"] = [stream_to_dict(x) for x in s.time]
    if s.strategy is not None:
        d = asdict(s.strategy)
        d["stop"] = list(d["stop"])
        out["strategy"] = d
    if s.quality is not None:
        out["quality"] = {k: v for k, v in asdict(s.quality).items() if v is not None}
    return out


def _diff_named(base: list[dict], new: list[dict]) -> list[dict]:
    old = {d["name"]: d for d in base}
    out = []
    for d in new:
        prev = old.get(d["name"])
        if prev is None:
            out.append(d)
            continue
        changed = {k: v for k, v in d.items() if prev.get(k) != v}
        # A field reset to its default disappears from the cleaned dict; write it explicitly.
        for k in prev:
            if k not in d:
                default = {**_STREAM_DEFAULTS, **_EVENT_DEFAULTS, "growth": {"mean": 0.0, "stdev": 0.0}}
                if k in default and k != "year":
                    changed[k] = default[k]
        if changed:
            out.append({"name": d["name"], **changed})
    names = {d["name"] for d in new}
    out += [{"name": n, "remove": True} for n in old if n not in names]
    return out


def _diff_map(base: dict, new: dict) -> dict:
    out = {}
    for k, v in new.items():
        b = base.get(k)
        if isinstance(v, dict) and isinstance(b, dict):
            sub = _diff_map(b, v)
            if sub:
                out[k] = sub
        elif b != v:
            out[k] = v
    return out


def diff(base: Scenario, new: Scenario) -> dict:
    """Override data that turns ``base`` into ``new`` (for an ``extends`` file)."""
    b, n = to_dict(base), to_dict(new)
    out: dict[str, Any] = {}
    for k, v in n.items():
        if k in ("income", "expenses", "events"):
            d = _diff_named(b.get(k, []), v)
            if d:
                out[k] = d
        elif isinstance(v, dict):
            d = _diff_map(b.get(k, {}), v)
            if d:
                out[k] = d
        elif b.get(k) != v:
            out[k] = v
    return out
