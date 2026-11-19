"""STATS Projections: a small Monte Carlo life planner.

Typical use from Python or a notebook:

    from stats_projections import load_scenario, simulate, summarize
    res = simulate(load_scenario("scenarios/baseline.yaml"))
    print(summarize(res).p_fail)
"""

from .config import Scenario, ScenarioError, load_scenario
from .engine import Result, simulate
from .report import compare, format_report, plot, summarize
from .sensitivity import Steps, analyze, format_sensitivity, plot_tornado

__all__ = [
    "Scenario", "ScenarioError", "load_scenario",
    "Result", "simulate",
    "compare", "format_report", "plot", "summarize",
    "Steps", "analyze", "format_sensitivity", "plot_tornado",
]
__version__ = "0.1.0"
