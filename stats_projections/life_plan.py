"""Life plans: a career, a place, a strategy and a timeline of life events at chosen ages.

A plan is a small JSON-friendly dict, so it can be saved and reloaded::

    {"name": "R&D, partner, two kids",
     "career": "rd_engineer", "location": None, "strategy": "work_to_65",
     "items": [{"kind": "partner", "age": 30},
               {"kind": "child", "age": 32}, {"kind": "child", "age": 35},
               {"kind": "home", "age": 34, "amount": 900000}]}

Timeline items reuse the life files where one exists (partner, children,
home, moving abroad), moved to the age you pick, and add simple events for
the rest (a big purchase, an inheritance, a career break, part-time work, an
earlier or later retirement). Ages are your age in the year the item starts.
``jitter`` is +- years of uncertainty in that timing.

``checks`` lists what-if variants of a plan (each item earlier, later or not
at all) to see which assumptions the plan leans on.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Scenario, ScenarioError, _merge, from_dict, load_raw

ROOT = Path(__file__).resolve().parent.parent
LIFE_DIR = ROOT / "life"

#: Timeline kinds: label, what "amount" means (None = not used), its default,
#: and whether "years" is used.
KINDS: dict[str, dict[str, Any]] = {
    "partner": {"label": "Partner moves in", "amount": "Partner pay ($/yr)", "default": 90_000},
    "child": {"label": "Child born", "amount": None},
    "home": {"label": "Buy a home", "amount": "Price ($)", "default": 1_100_000},
    "abroad": {"label": "Move abroad (remote work)", "amount": None},
    "purchase": {"label": "Big purchase", "amount": "Cost ($)", "default": 40_000},
    "windfall": {"label": "Inheritance or gift", "amount": "Amount ($)", "default": 100_000},
    "break": {"label": "Career break", "amount": None, "years": 1},
    "part_time": {"label": "Go part-time", "amount": "Share of full pay (%)", "default": 60},
    "retire": {"label": "Retire", "amount": None},
}
LABELS = {k: v["label"] for k, v in KINDS.items()}
KIND_OF = {v["label"]: k for k, v in KINDS.items()}
MAX_CHILDREN = 3
WORK = ["Career pay", "Work hours", "Employer 401(k) match", "TSP agency match"]


def _rename(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_rename(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rename(v, old, new) for k, v in obj.items()}
    return obj


def _event(raw: dict, name: str) -> dict:
    for e in raw.get("events", []):
        if e.get("name") == name:
            return e
    raise ScenarioError(f"life file has no event '{name}'")


def _streams(raw: dict) -> set[str]:
    return {s["name"] for k in ("income", "time", "expenses") for s in raw.get(k, [])}


def _item_part(item: dict, year: int, n_child: int, raw: dict, label: str) -> dict:
    """The raw dict an item adds on top of the plan so far."""
    kind = item["kind"]
    jitter = int(item.get("jitter") or 0)
    amount = item.get("amount")
    if kind == "partner":
        part = load_raw(LIFE_DIR / "partner.yaml")
        e = _event(part, "Partner moves in")
        e.update(year=year, jitter=jitter)
        if amount:
            e["amount"] = float(amount)
    elif kind == "child":
        part = _rename(load_raw(LIFE_DIR / "kids_1.yaml"), "Child 1", f"Child {n_child}")
        _event(part, f"Child {n_child} born").update(year=year, jitter=jitter)
    elif kind == "home":
        part = load_raw(LIFE_DIR / "buy_home.yaml")
        e = _event(part, "Buy a home")
        e.update(year=year, jitter=jitter)
        if amount:
            e["amount"] = float(amount)
    elif kind == "abroad":
        part = load_raw(LIFE_DIR / "expat_southern_europe.yaml")
        _event(part, "Move abroad").update(year=year, jitter=jitter)
        part["lifestyle"] = {"cost_index": [[year, 0.6]]}
    elif kind == "purchase":
        part = {"events": [{"name": label, "type": "purchase", "year": year, "jitter": jitter,
                            "amount": float(amount or KINDS[kind]["default"])}]}
    elif kind == "windfall":
        part = {"events": [{"name": label, "type": "windfall", "year": year, "jitter": jitter,
                            "amount": float(amount or KINDS[kind]["default"])}]}
    elif kind in ("break", "part_time"):
        targets = [s for s in WORK if s in _streams(raw)]
        if not targets:
            raise ScenarioError(f"{LABELS[kind]}: the career has no pay to change")
        e = {"name": label, "type": "scale", "target": targets, "year": year, "jitter": jitter}
        if kind == "break":
            e.update(factor=0.0, duration=max(int(item.get("years") or 1), 1))
        else:
            share = float(amount or KINDS[kind]["default"])
            e["factor"] = min(max(share / 100.0, 0.0), 1.0)
        part = {"events": [e]}
    else:
        raise ScenarioError(f"unknown timeline item '{kind}'")
    for k in ("name", "description", "sources", "assumptions"):
        part.pop(k, None)
    return part


def _retire(raw: dict, year: int) -> None:
    for e in raw.get("events", []):
        if e.get("name") in ("Retire", "Partner retires"):
            e["year"] = year
    for s in raw.get("income", []):
        if s.get("name") in ("Employer 401(k) match", "TSP agency match") and s.get("end_year"):
            s["end_year"] = year


def build(plan: dict, profile: Path, career_dir: Path, location_dir: Path,
          strategy_dir: Path) -> Scenario:
    """The scenario for one plan: profile + career + timeline + location + strategy."""
    raw = load_raw(profile)
    raw.pop("name", None)
    career = career_dir / f"{plan['career']}.yaml"
    raw = _merge(raw, load_raw(career))
    a0 = int(raw.get("start_age") or 0)
    horizon = int(raw.get("horizon_years") or 0)
    counts: dict[str, int] = {}
    retire_at = None
    for item in sorted(plan.get("items", []), key=lambda i: i.get("age", 0)):
        kind = item.get("kind")
        if kind not in KINDS:
            raise ScenarioError(f"unknown timeline item '{kind}'")
        age = int(item.get("age") or a0)
        year = age - a0
        if not 0 <= year < horizon:
            raise ScenarioError(f"{LABELS[kind]} at {age}: pick an age from {a0} to "
                                f"{a0 + horizon - 1}")
        if kind == "retire":
            retire_at = year
            continue
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "child" and counts[kind] > MAX_CHILDREN:
            raise ScenarioError(f"at most {MAX_CHILDREN} children")
        if kind in ("partner", "home", "abroad") and counts[kind] > 1:
            raise ScenarioError(f"only one '{LABELS[kind]}' per plan")
        label = f"{LABELS[kind]} at {age}" if kind not in ("partner", "child", "home", "abroad") \
            else LABELS[kind]
        if counts[kind] > 1 and kind not in ("child",):
            label += f" ({counts[kind]})"
        raw = _merge(raw, _item_part(item, year, counts.get("child", 0), raw, label))
    loc = plan.get("location")
    if loc and not counts.get("abroad"):
        part = load_raw(location_dir / f"{loc}.yaml")
        for k in ("name", "description", "sources", "assumptions"):
            part.pop(k, None)
        raw = _merge(raw, part)
    if plan.get("strategy"):
        part = load_raw(strategy_dir / f"{plan['strategy']}.yaml")
        for k in ("name", "description", "sources", "assumptions"):
            part.pop(k, None)
        raw = _merge(raw, part)
    if retire_at is not None:
        _retire(raw, retire_at)
    raw["name"] = plan.get("name") or plan["career"]
    return replace(from_dict(raw), source=career.resolve())


def describe(plan: dict) -> str:
    bits = []
    for item in sorted(plan.get("items", []), key=lambda i: i.get("age", 0)):
        k = item["kind"]
        s = f"{LABELS[k].lower()} at {item['age']}"
        if item.get("amount") and KINDS[k]["amount"]:
            unit = "%" if k == "part_time" else ""
            s += f" ({'' if unit else '$'}{float(item['amount']):,.0f}{unit})"
        if k == "break":
            s += f" for {int(item.get('years') or 1)} yr"
        bits.append(s)
    return "; ".join(bits) or "no life events"


def checks(plan: dict, shift: int = 3) -> list[tuple[str, dict]]:
    """What-if variants: each item ``shift`` years earlier, later, and left out."""
    out = []
    items = plan.get("items", [])
    seen: dict[str, int] = {}
    for i, item in enumerate(items):
        k = item["kind"]
        seen[k] = seen.get(k, 0) + 1
        name = LABELS[k] + (f" {seen[k]}" if k in ("child", "purchase", "windfall", "break",
                                                   "part_time") and
                            sum(x["kind"] == k for x in items) > 1 else "")
        for d, word in ((-shift, "earlier"), (shift, "later")):
            v = copy.deepcopy(plan)
            v["items"][i]["age"] = int(item["age"]) + d
            out.append((f"{name}: {shift} years {word}", v))
        v = copy.deepcopy(plan)
        v["items"].pop(i)
        out.append((f"{name}: doesn't happen" if k != "retire" else "Retire: at 65 as planned",
                    v))
    return out
