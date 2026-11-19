"""Command line entry point.

    python -m stats_projections scenarios/baseline.yaml scenarios/buy_home.yaml
    python -m stats_projections scenarios/*.yaml --chart outputs/compare.png --csv outputs
    python -m stats_projections scenarios/baseline.yaml --tornado outputs/tornado.png

The first scenario is the baseline the others are compared against.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from .config import ScenarioError, compose, load_scenario
from .engine import simulate
from .report import _slug, export_csv, format_milestones, format_report, plot
from .sensitivity import Steps, analyze, format_sensitivity, plot_tornado


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stats_projections",
        description="Monte Carlo life planner: how often do my finances fail?",
    )
    ap.add_argument("scenarios", nargs="+", help="scenario YAML files; the first is the baseline")
    ap.add_argument("--paths", type=int, help="override the number of paths")
    ap.add_argument("--seed", type=int, help="override the random seed")
    ap.add_argument("--chart", help="save a PNG chart to this path")
    ap.add_argument("--csv", help="write percentile CSVs into this folder")
    ap.add_argument("--profile", help="personal profile YAML; each file given is then a "
                    "career path layered on top of it (e.g. careers/*.yaml)")
    ap.add_argument("--add", action="append", default=[], metavar="FILE",
                    help="with --profile: a life-event or strategy file layered on every career "
                         "(repeatable), e.g. --add life/partner.yaml --add strategies/coast_fire.yaml")
    sens = ap.add_argument_group("sensitivity analysis")
    sens.add_argument("--sensitivity", action="store_true",
                      help="rank which inputs move each scenario's failure rate most")
    sens.add_argument("--tornado", help="save tornado chart(s); with several scenarios, "
                      "the scenario name is added to the file name")
    sens.add_argument("--top", type=int, default=15, help="inputs to show (default 15)")
    sens.add_argument("--money-step", type=float, default=0.10,
                      help="relative nudge for dollar amounts (default 0.10 = 10%%)")
    sens.add_argument("--rate-step", type=float, default=0.01,
                      help="absolute nudge for return and inflation means (default 0.01)")
    sens.add_argument("--year-step", type=int, default=2,
                      help="years to move event timing (default 2)")
    args = ap.parse_args(argv)
    if args.tornado:
        args.sensitivity = True

    files: list[str] = []
    for pattern in args.scenarios:  # expand globs ourselves; PowerShell does not
        matches = sorted(glob.glob(pattern))
        # Within a glob, put a baseline (or the R&D career) first as the reference.
        matches.sort(key=lambda f: 0 if any(k in f.lower() for k in ("baseline", "rd_engineer")) else 1)
        files += matches or [pattern]

    try:
        if args.profile:
            scenarios = [compose(args.profile, f, *args.add, paths=args.paths, seed=args.seed)
                         for f in files]
        else:
            scenarios = [load_scenario(f, paths=args.paths, seed=args.seed) for f in files]
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    keys = {(s.seed, s.paths, s.horizon_years) for s in scenarios}
    if len(keys) > 1:
        print(
            "error: compared scenarios must share seed, paths and horizon_years so they "
            "use the same random draws; found " + ", ".join(
                f"{s.name}=(seed {s.seed}, {s.paths} paths, {s.horizon_years} yrs)"
                for s in scenarios
            ),
            file=sys.stderr,
        )
        return 2

    t0 = time.perf_counter()
    results = [simulate(s) for s in scenarios]
    elapsed = time.perf_counter() - t0

    print(format_report(results))
    if args.profile:
        print(format_milestones(results))
    print(f"\n({len(results)} scenario(s) simulated in {elapsed:.2f}s)")
    if args.chart:
        print(f"chart: {plot(results, args.chart)}")
    if args.csv:
        for p in export_csv(results, args.csv):
            print(f"csv:   {p}")

    if args.sensitivity:
        steps = Steps(money=args.money_step, rate=args.rate_step, years=args.year_step)
        for res in results:
            t0 = time.perf_counter()
            sa = analyze(res.scenario, steps, base=res)
            print("\n" + format_sensitivity(sa, top=args.top))
            print(f"({2 * len(sa.rows)} extra runs in {time.perf_counter() - t0:.1f}s)")
            if args.tornado:
                path = Path(args.tornado)
                if len(results) > 1:
                    path = path.with_name(f"{path.stem}_{_slug(res.scenario.name)}{path.suffix}")
                print(f"tornado: {plot_tornado(sa, path, top=args.top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
