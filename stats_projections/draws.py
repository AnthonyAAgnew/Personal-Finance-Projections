"""Random draws, keyed so scenarios share them (common random numbers).

Every source of randomness gets its own generator seeded from the scenario
seed plus a stable hash of a key such as ``("growth", "Salary")``. Two
scenarios with the same seed, path count, and horizon therefore see exactly
the same market years, the same raises for a stream of the same name, and the
same job-gap years. Their difference reflects the decision, not noise.
"""

from __future__ import annotations

import csv
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Inflation, Returns, ScenarioError


def rng_for(seed: int, *key: str) -> np.random.Generator:
    tag = zlib.crc32("/".join(key).encode("utf-8"))
    return np.random.default_rng([seed, tag])


def standard_normal(seed: int, paths: int, years: int, *key: str) -> np.ndarray:
    return rng_for(seed, *key).standard_normal((paths, years))


def uniform(seed: int, paths: int, years: int, *key: str) -> np.ndarray:
    return rng_for(seed, *key).random((paths, years))


@lru_cache(maxsize=8)
def load_history(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV with columns year, return, inflation (nominal decimals)."""
    p = Path(path)
    if not p.exists():
        raise ScenarioError(f"bootstrap data not found: {p}")
    rets, infl = [], []
    with p.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"return", "inflation"} - set(reader.fieldnames or [])
        if missing:
            raise ScenarioError(f"{p.name}: missing column(s) {sorted(missing)}")
        for row in reader:
            rets.append(float(row["return"]))
            infl.append(float(row["inflation"]))
    if len(rets) < 2:
        raise ScenarioError(f"{p.name}: needs at least two years of data")
    r, i = np.array(rets), np.array(infl)
    if np.abs(r).max() > 2 or np.abs(i).max() > 2:
        raise ScenarioError(f"{p.name}: values look like percents; use decimals (0.07, not 7)")
    return r, i


def market(
    returns: Returns, inflation: Inflation, seed: int, paths: int, years: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal returns and inflation, each shaped (paths, years)."""
    if returns.model == "bootstrap":
        hist_r, hist_i = load_history(returns.data)
        n, b = len(hist_r), min(returns.block_size, len(hist_r))
        blocks = -(-years // b)
        starts = rng_for(seed, "market", "bootstrap").integers(0, n - b + 1, (paths, blocks))
        idx = (starts[:, :, None] + np.arange(b)).reshape(paths, -1)[:, :years]
        return hist_r[idx], hist_i[idx]

    z_r = standard_normal(seed, paths, years, "market", "returns")
    if returns.model == "student_t":
        # Scaled t: same mean and stdev as the normal model, fatter tails.
        dof = returns.dof
        chi = rng_for(seed, "market", "t-scale").chisquare(dof, (paths, years))
        z_r = z_r / np.sqrt(chi / dof) * np.sqrt((dof - 2) / dof)
    nominal = returns.mean + returns.stdev * z_r

    z_i = standard_normal(seed, paths, years, "market", "inflation")
    rho = inflation.correlation  # z_r has unit variance under both models
    z_i = rho * z_r + np.sqrt(1 - rho**2) * z_i
    infl = inflation.mean + inflation.stdev * z_i
    return np.maximum(nominal, -0.99), np.maximum(infl, -0.5)
