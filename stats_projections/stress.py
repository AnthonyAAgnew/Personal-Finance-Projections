"""Market stress tests: named sequences of bad years laid over the random draws.

Each preset is a tuple of (year, stock return delta, inflation delta) added to
every path's draws in those years, so the comparison stays paired: the same
random markets, with the same shock on top. Deltas are nominal and additive
(-0.40 turns a +7% year into -33%).

The shocks are illustrative, loosely sized on US history:
* Lost decade: stocks returned roughly 0% nominal over 2000-2009.
* High inflation: 1973-1982 inflation averaged ~8.7% with weak real returns.
* Crashes: the S&P 500 fell ~37% in 2008 and ~43% in 1931.
"""

from __future__ import annotations

Stress = tuple[tuple[int, float, float], ...]


def presets(career_start: int, retire_year: int) -> dict[str, Stress]:
    cs, ry = career_start, retire_year
    return {
        "Long-run expected (no stress)": (),
        "Lost decade at career start": tuple((y, -0.06, 0.0) for y in range(cs, cs + 10)),
        "High inflation decade (1970s-style)": tuple((y, -0.03, 0.05) for y in range(cs, cs + 10)),
        "Crash in your first career year": ((cs, -0.40, 0.0),),
        "Crash right after retiring": ((ry, -0.40, 0.0), (ry + 1, -0.10, 0.0)),
        "Depression-level shock (age ~35)": ((cs + 9, -0.35, -0.03), (cs + 10, -0.30, -0.05),
                                             (cs + 11, -0.15, -0.02)),
    }


DESCRIPTIONS = {
    "Long-run expected (no stress)": "The return model as set, with no extra shocks.",
    "Lost decade at career start": "Stocks about 6 points a year below normal for your first ten "
                                   "career years (roughly 0% nominal, like 2000-2009).",
    "High inflation decade (1970s-style)": "Inflation 5 points higher and stocks 3 points lower "
                                           "for your first ten career years.",
    "Crash in your first career year": "Stocks fall about 33% the year your career starts.",
    "Crash right after retiring": "Stocks fall about 33% the year you retire and lag the next "
                                  "year: the classic sequence-of-returns risk.",
    "Depression-level shock (age ~35)": "Three bad years in a row (about -28%, -23%, -8%) with "
                                        "deflation, around age 35.",
}
