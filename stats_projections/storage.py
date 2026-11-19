"""Saved plans, result snapshots and net-worth check-ins.

Three kinds of records, all small JSON-friendly dicts:

* **plans**: a named setup (what to compare, simulation settings and any
  what-if edits), so the same analysis can be reopened on another device;
* **snapshots**: dated key results for a plan, so you can see how the outlook
  moves as assumptions and your situation change;
* **check-ins**: your actual net worth on a date, compared with what an earlier
  snapshot projected.

Two backends share one interface:

* ``LocalStore``: a folder on your own computer (``saved/``, git-ignored).
* ``GistStore``: one *secret* GitHub gist, for the hosted app. Needs a GitHub
  token with gist access and the gist's id (Streamlit secrets, see README).
  A secret gist is unlisted, not encrypted: anyone with its URL can read it,
  so never share the URL.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

PLAN_PREFIX = "plan__"
SNAPSHOTS = "snapshots.json"
CHECKINS = "checkins.json"


class StorageError(RuntimeError):
    pass


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "plan"


class Store:
    """Files are named strings of JSON text. Subclasses implement the 4 primitives."""

    label = "storage"

    # -- primitives -----------------------------------------------------
    def _list(self) -> list[str]:
        raise NotImplementedError

    def _read(self, name: str) -> str | None:
        raise NotImplementedError

    def _write(self, files: dict[str, str | None]) -> None:
        """Write (or delete, with None) several files at once."""
        raise NotImplementedError

    # -- helpers --------------------------------------------------------
    def _json(self, name: str, default: Any) -> Any:
        text = self._read(name)
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise StorageError(f"{name} is not valid JSON: {exc}") from exc

    @staticmethod
    def _dump(obj: Any) -> str:
        return json.dumps(obj, indent=1, sort_keys=False, default=str)

    # -- plans ----------------------------------------------------------
    def plans(self) -> dict[str, dict]:
        """Saved plans by display name, newest save first."""
        out = []
        for f in self._list():
            if f.startswith(PLAN_PREFIX) and f.endswith(".json"):
                p = self._json(f, None)
                if isinstance(p, dict) and p.get("name"):
                    out.append(p)
        out.sort(key=lambda p: p.get("saved", ""), reverse=True)
        return {p["name"]: p for p in out}

    def save_plan(self, plan: dict, snapshot: dict | None = None) -> None:
        plan = {**plan, "saved": plan.get("saved") or date.today().isoformat()}
        files: dict[str, str | None] = {f"{PLAN_PREFIX}{slug(plan['name'])}.json": self._dump(plan)}
        if snapshot is not None:
            snaps = self.snapshots()
            snaps.append(snapshot)
            files[SNAPSHOTS] = self._dump(snaps)
        self._write(files)

    def delete_plan(self, name: str) -> None:
        self._write({f"{PLAN_PREFIX}{slug(name)}.json": None})

    # -- snapshots --------------------------------------------------------
    def snapshots(self) -> list[dict]:
        s = self._json(SNAPSHOTS, [])
        return s if isinstance(s, list) else []

    def add_snapshot(self, snapshot: dict) -> None:
        snaps = self.snapshots()
        snaps.append(snapshot)
        self._write({SNAPSHOTS: self._dump(snaps)})

    def delete_snapshot(self, index: int) -> None:
        snaps = self.snapshots()
        if 0 <= index < len(snaps):
            snaps.pop(index)
            self._write({SNAPSHOTS: self._dump(snaps)})

    # -- check-ins --------------------------------------------------------
    def checkins(self) -> list[dict]:
        c = self._json(CHECKINS, [])
        return sorted(c, key=lambda x: x.get("date", "")) if isinstance(c, list) else []

    def add_checkin(self, checkin: dict) -> None:
        c = [x for x in self.checkins() if x.get("date") != checkin.get("date")]
        c.append(checkin)
        c.sort(key=lambda x: x.get("date", ""))
        self._write({CHECKINS: self._dump(c)})

    def delete_checkin(self, day: str) -> None:
        self._write({CHECKINS: self._dump([x for x in self.checkins() if x.get("date") != day])})


class LocalStore(Store):
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.label = f"this computer ({self.folder.name}/ folder)"

    def _list(self) -> list[str]:
        return sorted(p.name for p in self.folder.glob("*.json")) if self.folder.exists() else []

    def _read(self, name: str) -> str | None:
        p = self.folder / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _write(self, files: dict[str, str | None]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            p = self.folder / name
            if text is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(text, encoding="utf-8")


class GistStore(Store):
    """One secret gist. Reads are cached until the next write."""

    API = "https://api.github.com/gists/"

    def __init__(self, token: str, gist_id: str, timeout: float = 15.0):
        self.token, self.gist_id, self.timeout = token, gist_id.strip(), timeout
        self.label = "your private GitHub gist"
        self._cache: dict[str, str] | None = None

    def _request(self, method: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            self.API + self.gist_id, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}",
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28",
                     "User-Agent": "stats-projections"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            hint = {401: "the token is wrong or expired",
                    403: "the token lacks gist access",
                    404: "no gist with that id (or the token can't see it)"}.get(exc.code, "")
            raise StorageError(f"GitHub returned {exc.code}{': ' + hint if hint else ''}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise StorageError(f"Could not reach GitHub: {exc}") from exc

    def _files(self) -> dict[str, str]:
        if self._cache is None:
            data = self._request("GET")
            files = {}
            for name, f in data.get("files", {}).items():
                if f.get("truncated"):
                    raise StorageError(f"{name} is too large for the gist API")
                files[name] = f.get("content") or ""
            self._cache = files
        return self._cache

    def _list(self) -> list[str]:
        return sorted(self._files())

    def _read(self, name: str) -> str | None:
        return self._files().get(name)

    def _write(self, files: dict[str, str | None]) -> None:
        existing = self._files()
        # Deleting a file that isn't there is an error on GitHub's side.
        body = {n: (None if t is None else {"content": t}) for n, t in files.items()
                if t is not None or n in existing}
        if body:
            self._request("PATCH", {"files": body})
        self._cache = None


# ----------------------------------------------------------------------------
# Comparing a check-in with an earlier projection
# ----------------------------------------------------------------------------

def years_between(a: str, b: str) -> float:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days / 365.25


def to_todays_dollars(amount: float, years: float, inflation: float) -> float:
    """Nominal dollars ``years`` after the snapshot, in the snapshot's dollars."""
    return amount / (1 + inflation) ** max(years, 0.0)


def percentile_of(value: float, band: dict[str, list[float]], years: float) -> float | None:
    """Rough percentile of ``value`` within a stored projection band at ``years``.

    ``band`` holds the 10/25/50/75/90th percentiles by whole year from the
    snapshot. Interpolates in time, then linearly between the stored
    percentiles (clamped to 5 and 95 outside the band).
    """
    qs = [10, 25, 50, 75, 90]
    rows = [band.get(f"p{q}") for q in qs]
    if any(r is None for r in rows):
        return None
    n = len(rows[0])
    if years < 0 or years > n - 1:
        return None
    lo = int(years)
    hi = min(lo + 1, n - 1)
    w = years - lo
    at = [r[lo] * (1 - w) + r[hi] * w for r in rows]
    if value <= at[0]:
        return 5.0 if value < at[0] else 10.0
    if value >= at[-1]:
        return 95.0 if value > at[-1] else 90.0
    for (q0, v0), (q1, v1) in zip(zip(qs, at), zip(qs[1:], at[1:])):
        if v0 <= value <= v1:
            return q0 + (q1 - q0) * (0.0 if v1 == v0 else (value - v0) / (v1 - v0))
    return None
