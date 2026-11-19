"""STATS Projections: Streamlit front end over the same engine as the CLI.

    streamlit run app.py

Two modes:

* Careers: a personal profile (profiles/) plus each career path (careers/),
  compared on the same random draws.
* Life scenarios: the standalone scenario files in scenarios/.

Files stay the source of truth. The "What if" tab edits a copy of the
reference and can save the edits as a small override file that ``extends`` it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml


def _fresh_engine() -> None:
    """Re-import stats_projections when its source files change.

    Streamlit reruns this script on every change but keeps imported modules,
    so a redeploy that updates both the YAML files and the engine would
    otherwise run new files against old code (Streamlit Cloud does this).
    """
    pkg = Path(__file__).resolve().parent / "stats_projections"
    stamp = tuple(sorted((p.name, p.stat().st_mtime_ns) for p in pkg.glob("*.py")))
    mod = sys.modules.get("stats_projections")
    if mod is not None and getattr(mod, "_source_stamp", None) != stamp:
        for name in [n for n in sys.modules if n.split(".")[0] == "stats_projections"]:
            del sys.modules[name]
        st.cache_resource.clear()
    import stats_projections
    stats_projections._source_stamp = stamp


_fresh_engine()

from stats_projections import ScenarioError, compare, load_scenario, simulate, summarize
from stats_projections.config import career_info, compose, from_dict
from stats_projections.edit import diff, to_dict
from stats_projections.report import (
    COLORS, INCOME_MILESTONES, NET_WORTH_MILESTONES, WEEKS_PER_YEAR, _slug, career_metrics,
    debt_metrics, hours_metrics, income_milestones, net_worth_milestones, strategy_metrics,
    tier_shares)
from stats_projections.decisions import RULES as DP_RULES, decide as dp_decide
from stats_projections.decide import CRITERIA, DEFAULT_WEIGHTS, pareto, score
from stats_projections.sensitivity import Steps, analyze, knobs
from stats_projections.breakeven import METRICS as BE_METRICS, levers as be_levers, solve as be_solve
from stats_projections.stress import DESCRIPTIONS as STRESS_HELP, presets as stress_presets
from stats_projections.report import (FAILURE_MODES, accounts_at, failure_modes, free_time,
                                      runway_metrics)

ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "scenarios"
CAREER_DIR = ROOT / "careers"
PROFILE_DIR = ROOT / "profiles"
LIFE_DIR = ROOT / "life"
LOCATION_DIR = ROOT / "locations"
STRATEGY_DIR = ROOT / "strategies"
WHAT_IF = "What if"

st.set_page_config(page_title="STATS Projections", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# Phone layout: tighter margins, smaller headings and tabs that scroll sideways.
st.markdown("""<style>
@media (max-width: 640px) {
  .block-container {padding: 2.5rem 0.75rem 3rem !important;}
  h1 {font-size: 1.6rem !important;}
  h2 {font-size: 1.3rem !important;}
  h3 {font-size: 1.1rem !important;}
  [data-testid="stMetricValue"] {font-size: 1.35rem;}
  [data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 0.4rem; overflow-x: auto;
      scrollbar-width: none;}
  [data-testid="stTabs"] button[role="tab"] {padding: 0.2rem 0.35rem; white-space: nowrap;}
  [data-testid="stTabs"] button[role="tab"] p {font-size: 0.92rem;}
  [data-testid="stExpander"] summary p {font-size: 1rem;}
}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached engine calls. Scenario is a frozen dataclass, so it is a dict key.
# ----------------------------------------------------------------------------


@st.cache_resource
def _cache() -> dict:
    return {}


def run(s):
    cache = _cache()
    if s not in cache:
        if len(cache) > 64:
            cache.clear()
        cache[s] = simulate(s)
    return cache[s]


def run_sensitivity(s, steps: Steps):
    cache = _cache()
    key = ("sensitivity", s, steps)
    if key not in cache:
        cache[key] = analyze(s, steps, base=run(s))
    return cache[key]


def load_all(mode: str, profile: Path | None = None, life=(), strategy: Path | None = None,
             career: Path | None = None) -> dict[str, tuple[Path, object]]:
    """The options to compare in each mode.

    * Careers: every career file on the profile, plus life events and one strategy.
    * Strategies: one career on the profile, plus life events, under every strategy.
    * Life scenarios: the standalone files in scenarios/.
    """
    out, errors = {}, []
    if mode == "Careers":
        items = [(f, (profile, f, *life, *([strategy] if strategy else [])), {})
                 for f in sorted(CAREER_DIR.glob("*.yaml"))]
    elif mode == "Strategies":
        items = [(g, (profile, career, *life, g), {"name": file_name(g)})
                 for g in sorted(STRATEGY_DIR.glob("*.yaml"))]
    else:
        items = [(f, None, {}) for f in sorted(SCENARIO_DIR.glob("*.yaml"))]
    for f, parts, over in items:
        try:
            s = compose(*parts, **over) if parts else load_scenario(f)
            out[s.name] = (f, s)
        except ScenarioError as exc:
            errors.append(f"{f.name}: {exc}")
    for e in errors:
        st.error(e)
    return out


def file_name(f: Path) -> str:
    try:
        return career_info(f)["name"]
    except ScenarioError:
        return f.stem


def show_chart(chart, **kwargs):
    """Render an Altair chart with its legend underneath, so phones keep the full width."""
    chart = chart.configure_legend(orient="bottom", direction="vertical", columns=1,
                                   symbolOpacity=1, symbolStrokeWidth=3,
                                   labelLimit=220, title=None)
    # Fit the width to the screen but keep the plot's own height; the legend
    # below then adds to the height instead of squashing the plot.
    chart = chart.properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    st.altair_chart(chart, **kwargs)


def money(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    return f"{sign}${x / 1e3:,.0f}k" if x >= 1e3 else f"{sign}${x:,.0f}"


# ----------------------------------------------------------------------------
# Sidebar: pick what to compare and the shared assumptions
# ----------------------------------------------------------------------------

CORE_CAREERS = ("rd_engineer", "regulator", "device_sales", "doctor", "cardiology_ep", "entrepreneur")

st.title("STATS Projections")
summary_slot = st.empty()
setup = st.expander("⚙️ Setup: what to compare", expanded=False)
sim = st.expander("🎲 Simulation settings", expanded=False)
with setup:
    setup_left, setup_right = st.columns(2)

with setup_left:
    mode = st.radio("Compare", ["Careers", "Strategies", "Life scenarios"], horizontal=True,
                    help="Careers: your profile under each career path. Strategies: one career "
                         "under each finance strategy (FIRE, coast...). Life scenarios: the "
                         "standalone files in scenarios/.")
    profile_file, life_files, strategy_file, career_file = None, [], None, None
    location_file = None
    if mode != "Life scenarios":
        profiles = sorted(PROFILE_DIR.glob("*.yaml"))
        if not profiles:
            st.error(f"No profile files in {PROFILE_DIR}")
            st.stop()
        profile_file = st.selectbox("Profile", profiles, format_func=lambda p: p.stem)
        if mode == "Strategies":
            careers = sorted(CAREER_DIR.glob("*.yaml"))
            career_file = st.selectbox(
                "Career", careers, format_func=file_name,
                index=next((i for i, c in enumerate(careers) if c.stem == "rd_engineer"), 0))
        life_files = st.multiselect("Life events", sorted(LIFE_DIR.glob("*.yaml")),
                                    format_func=file_name,
                                    help="Layered on every option: partner, kids, moving abroad.")
        location_file = st.selectbox(
            "Location after the Ph.D.", [None, *sorted(LOCATION_DIR.glob("*.yaml"))],
            format_func=lambda f: "Irvine (profile)" if f is None else file_name(f),
            help="Pay, living costs and state taxes relative to Irvine, from the start of "
                 "your career. Applied after life events, so a partner's joint return keeps "
                 "the new state's taxes.")
        if location_file and any(f.stem.startswith("expat") for f in life_files):
            st.warning("Moving abroad replaces the US location, so the location is ignored.")
            location_file = None
        if location_file:
            life_files = [*life_files, location_file]
        if mode == "Careers":
            strategies = sorted(STRATEGY_DIR.glob("*.yaml"))
            strategy_file = st.selectbox(
                "Finance strategy", strategies, format_func=file_name,
                index=next((i for i, g in enumerate(strategies) if g.stem == "work_to_65"), 0))

scenarios = load_all(mode, profile_file, tuple(life_files), strategy_file, career_file)
if not scenarios:
    st.error("No valid scenario or career files found.")
    st.stop()

names = list(scenarios)
ref_key = {"Careers": "rd_engineer", "Strategies": "work_to_65"}.get(mode, "baseline")
default_base = next((n for n, (f, _) in scenarios.items() if f.stem == ref_key), names[0])

with setup_right:
    label = {"Careers": "Reference career", "Strategies": "Reference strategy"}.get(mode, "Baseline")
    base_name = st.selectbox(label, names, index=names.index(default_base))
    others = [n for n in names if n != base_name]
    if mode == "Careers":
        default_others = [n for n in others if scenarios[n][0].stem in CORE_CAREERS]
        st.caption(f"{len(names)} careers available; the six core paths are preselected.")
    else:
        default_others = others
    chosen = st.multiselect("Compare with", others, default=default_others)

    base_file, base = scenarios[base_name]
    save_dir = {"Careers": CAREER_DIR, "Life scenarios": SCENARIO_DIR}.get(mode)


with sim:
    st.caption("Applied to every option, so comparisons stay fair.")
    paths = st.select_slider("Paths", [1000, 2000, 5000, 10000, 20000, 50000], value=base.paths)
    seed = st.number_input("Random seed", value=base.seed, step=1)
    floor = st.number_input("Failure floor ($)", value=float(base.failure_floor), step=5000.0)

    r = base.returns
    if r.model == "bootstrap":
        st.info(f"Returns resampled from {Path(r.data).name} (block {r.block_size}).")
        returns = r
    else:
        model = st.radio("Return model", ["normal", "student_t"],
                         index=0 if r.model == "normal" else 1, horizontal=True,
                         help="student_t keeps the mean and volatility but adds fatter tails (more crashes).")
        mean = st.slider("Stock return (nominal)" if base.allocation else "Portfolio return (nominal)",
                         0.0, 12.0, round(r.mean * 100, 2), 0.25, format="%.2f%%",
                         help="With a glide path (profile allocation), this is the stock part; "
                              "bonds and cash have their own returns in the profile.")
        stdev = st.slider("Return volatility", 0.0, 30.0, round(r.stdev * 100, 1), 0.5,
                          format="%.1f%%")
        dof = r.dof
        if model == "student_t":
            dof = st.slider("Tail heaviness (degrees of freedom)", 3.0, 30.0, float(r.dof), 0.5,
                            help="Lower = fatter tails.")
        returns = replace(r, model=model, mean=round(mean / 100, 6),
                          stdev=round(stdev / 100, 6), dof=dof)

    i = base.inflation
    infl_mean = st.slider("Inflation", 0.0, 8.0, round(i.mean * 100, 1), 0.1, format="%.1f%%")
    infl_sd = st.slider("Inflation volatility", 0.0, 5.0, round(i.stdev * 100, 1), 0.1,
                        format="%.1f%%")
    inflation = replace(i, mean=round(infl_mean / 100, 6), stdev=round(infl_sd / 100, 6))

    retire_ev = next((e for e in base.events if e.name == "Retire"), None)
    retire_year = retire_ev.year if retire_ev else max(base.horizon_years - 30, 1)
    career_start = next((x.start_year for x in base.income if x.name == "Career pay"), 0)
    stress_options = stress_presets(career_start, retire_year)
    stress_name = st.selectbox("Market scenario", list(stress_options),
                               help="Stress tests lay a bad stretch of years over every random "
                                    "path, so options are still compared on the same markets.")
    st.caption(STRESS_HELP[stress_name])
    stress = stress_options[stress_name]


loc_label = "Irvine" if not location_file else file_name(location_file)
summary_bits = [f"**{base_name}** vs {len(chosen)} other{'s' if len(chosen) != 1 else ''}"]
if mode != "Life scenarios":
    summary_bits.append(loc_label)
    if strategy_file is not None:
        summary_bits.append(file_name(strategy_file))
    elif career_file is not None:
        summary_bits.append(file_name(career_file))
    if life_files:
        extra = [f for f in life_files if f != location_file]
        if extra:
            summary_bits.append(", ".join(file_name(f) for f in extra))
summary_bits += [f"{int(paths):,} paths", stress_name]
summary_slot.caption(f"{mode}: " + " · ".join(summary_bits))


def shared(s):
    return replace(s, paths=int(paths), seed=int(seed), failure_floor=float(floor),
                   returns=returns, inflation=inflation, stress=stress or s.stress)


# ----------------------------------------------------------------------------
# What-if editor (built before the results so they can include it)
# ----------------------------------------------------------------------------

tab_names = ["Results", "Milestones", "Lifestyle", "Hours", "Cash flow", "What if",
             "Sensitivity", "Files"]
if profile_file:
    tab_names.insert(5, "Assumptions")
    tab_names.insert(1, "Decide")
TAB_GROUPS = {
    "Overview": ["Results", "Milestones"],
    "Compare": ["Decide", "Sensitivity"],
    "Plan": ["What if", "Lifestyle", "Hours", "Cash flow"],
    "Details": ["Assumptions", "Files"],
}
tabs = {}
for group, top in zip(TAB_GROUPS, st.tabs(list(TAB_GROUPS))):
    subs = [n for n in TAB_GROUPS[group] if n in tab_names]
    with top:
        if len(subs) == 1:
            tabs[subs[0]] = st.container()
        else:
            tabs.update(zip(subs, st.tabs(subs)))
tab_results, tab_edit, tab_sens, tab_files = (
    tabs["Results"], tabs["What if"], tabs["Sensitivity"], tabs["Files"])

base_raw = to_dict(base)
prev_streams = {d["name"]: d for d in (*base_raw["income"], *base_raw["expenses"])}
prev_events = {d["name"]: d for d in base_raw["events"]}
# Editor state belongs to one reference and one set of add-ons.
ed = f"{mode}/{base_name}/{[p.stem for p in life_files]}/{getattr(strategy_file, 'stem', '')}"

with tab_edit:
    st.markdown(
        f"Edit a copy of **{base_name}**. Changes appear as a **{WHAT_IF}** line on the "
        "Results tab. Money is in today's dollars. Year 0 is now. Growth is real "
        "(above inflation). Taxable income is gross pay."
    )
    if st.button("Reset to reference"):
        for k in list(st.session_state):
            if str(k).startswith(ed):
                del st.session_state[k]

    start_balance = st.number_input(
        "Starting balance ($)", value=float(base.starting_balance), step=5000.0, key=f"{ed}/start"
    )

    def stream_frame(items):
        return pd.DataFrame([{
            "name": x.name, "amount": x.amount, "start_year": x.start_year,
            "end_year": x.end_year, "growth %": round(x.growth.mean * 100, 8),
            "growth volatility %": round(x.growth.stdev * 100, 8),
            "noise %": round(x.noise * 100, 8),
            "job-gap chance %": round(x.loss_probability * 100, 8),
            "taxable": x.taxable,
        } for x in items])

    stream_cols = {
        "name": st.column_config.TextColumn("Name", required=True),
        "amount": st.column_config.NumberColumn("Amount ($/yr)", min_value=0, step=1000, format="%d"),
        "start_year": st.column_config.NumberColumn("Start year", min_value=0, step=1),
        "end_year": st.column_config.NumberColumn("End year", min_value=0, step=1,
                                                  help="First year without it. Blank = forever."),
        "growth %": st.column_config.NumberColumn("Growth %", step=0.1, format="%.2f"),
        "growth volatility %": st.column_config.NumberColumn("Growth vol %", min_value=0,
                                                             step=0.1, format="%.2f"),
        "noise %": st.column_config.NumberColumn("Noise %", min_value=0, step=1, format="%.0f",
                                                 help="Year-to-year swing that does not compound, "
                                                      "e.g. bonus or commission."),
        "job-gap chance %": st.column_config.NumberColumn("Job-gap %", min_value=0,
                                                          max_value=100, step=0.5, format="%.1f"),
        "taxable": st.column_config.CheckboxColumn("Gross (taxed)"),
    }

    st.subheader("Income")
    inc_df = st.data_editor(stream_frame(base.income), column_config=stream_cols,
                            num_rows="dynamic", hide_index=True, key=f"{ed}/income",
                            width="stretch")
    st.subheader("Expenses")
    exp_df = st.data_editor(stream_frame(base.expenses),
                            column_config={k: v for k, v in stream_cols.items() if k != "taxable"},
                            column_order=[c for c in stream_cols if c != "taxable"],
                            num_rows="dynamic", hide_index=True, key=f"{ed}/expenses",
                            width="stretch")

    stream_names = [n for n in [*inc_df["name"].tolist(), *exp_df["name"].tolist()]
                    if isinstance(n, str) and n]
    st.subheader("Events")
    st.caption(
        "purchase / windfall: one-time amount. set_amount: from that year the target stream "
        "becomes the amount (0 ends it); with a duration it reverts afterwards. loan: cash now, "
        "repaid after the deferral. Probability: chance it happens at all. Events in the same "
        "group are mutually exclusive outcomes. Linked events follow another event's year and "
        "only happen if it does."
    )
    ev_df = pd.DataFrame([{
        "name": e.name, "type": e.type, "year": e.year if not e.same_time_as else None,
        "amount": e.amount, "target": e.target, "probability %": round(e.probability * 100, 8),
        "group": e.group, "jitter": e.jitter, "same_time_as": e.same_time_as,
        "offset": e.offset, "duration": e.duration,
    } for e in base.events])
    event_names = [e.name for e in base.events]
    ev_df = st.data_editor(
        ev_df,
        column_config={
            "name": st.column_config.TextColumn("Name", required=True),
            "type": st.column_config.SelectboxColumn(
                "Type", options=["purchase", "windfall", "set_amount", "loan"], required=True),
            "year": st.column_config.NumberColumn("Year", min_value=0, step=1),
            "amount": st.column_config.NumberColumn("Amount ($)", min_value=0, step=1000, format="%d"),
            "target": st.column_config.SelectboxColumn("Target stream", options=stream_names),
            "probability %": st.column_config.NumberColumn("Probability %", min_value=0,
                                                           max_value=100, step=5, format="%.0f"),
            "group": st.column_config.TextColumn("Outcome group"),
            "jitter": st.column_config.NumberColumn("Jitter (yrs)", min_value=0, step=1),
            "same_time_as": st.column_config.SelectboxColumn("Linked to", options=event_names),
            "offset": st.column_config.NumberColumn("Offset (yrs)", step=1),
            "duration": st.column_config.NumberColumn("Duration (yrs)", min_value=1, step=1),
        },
        num_rows="dynamic", hide_index=True, key=f"{ed}/events", width="stretch",
    )
    st.caption("Loan rate, term and deferral, windfall tax and spread are kept from the file; "
               "edit them in the YAML.")

    ls0 = base.lifestyle
    tier_df = None
    if ls0 is not None:
        st.subheader("Lifestyle tiers (monthly budget, today's $)")
        st.caption("Each year your affordable income = 3-year average take-home + the withdrawal "
                   "rate x your portfolio. Reaching a tier's take-home switches to its budget; "
                   "between tiers you also spend the creep share of every extra dollar.")
        lc1, lc2 = st.columns(2)
        creep_in = lc1.number_input("Lifestyle creep (% of each extra dollar spent)", 0.0, 100.0,
                                    round(ls0.creep * 100, 6), 5.0, key=f"{ed}/creep")
        wr_in = lc2.number_input("Perpetual withdrawal rate (%)", 0.0, 10.0,
                                 round(ls0.withdrawal_rate * 100, 6), 0.25, key=f"{ed}/wr")
        cat_cols = list(ls0.categories)
        tier_df = st.data_editor(
            pd.DataFrame([{"name": t.name, "take_home": t.take_home,
                           **{c: dict(t.monthly).get(c) for c in cat_cols}} for t in ls0.tiers]),
            column_config={
                "name": st.column_config.TextColumn("Tier", required=True),
                "take_home": st.column_config.NumberColumn("Starts at take-home ($/yr)", min_value=0,
                                                           step=10000, format="%d"),
                **{c: st.column_config.NumberColumn(f"{c} ($/mo)", min_value=0, step=50, format="%d")
                   for c in cat_cols},
            },
            num_rows="dynamic", hide_index=True, key=f"{ed}/tiers", width="stretch",
        )
        monthly_totals = tier_df[cat_cols].fillna(0).sum(axis=1)
        st.caption("Monthly totals: " + ", ".join(
            f"{n}: ${v:,.0f}/mo (${12 * v:,.0f}/yr)"
            for n, v in zip(tier_df["name"], monthly_totals) if isinstance(n, str)))


def _num(v, default=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def pct(v) -> float:
    """Percent cell to a decimal, rounded so an untouched value round-trips exactly."""
    return round(float(_num(v, 0.0)) / 100, 10)


def rows_to_streams(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict("records"):
        if not isinstance(row.get("name"), str) or not row["name"].strip():
            continue
        name = row["name"].strip()
        # Start from the file's version so fields not shown in the table survive.
        d = dict(prev_streams.get(name, {"loss_fraction": 0.5}))
        d.update({
            "name": name,
            "amount": float(_num(row.get("amount"), 0.0)),
            "start_year": int(_num(row.get("start_year"), 0)),
            "growth": {"mean": pct(row.get("growth %")),
                       "stdev": pct(row.get("growth volatility %"))},
            "noise": pct(row.get("noise %")),
            "loss_probability": pct(row.get("job-gap chance %")),
            "taxable": bool(_num(row.get("taxable"), False)),
        })
        end = _num(row.get("end_year"))
        d["end_year"] = None if end is None else int(end)
        out.append(d)
    return out


def rows_to_events(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict("records"):
        if not isinstance(row.get("name"), str) or not row["name"].strip():
            continue
        name = row["name"].strip()
        d = dict(prev_events.get(name, {}))
        d.update({"name": name, "type": row.get("type") or "purchase",
                  "amount": float(_num(row.get("amount"), 0.0)),
                  "probability": pct(_num(row.get("probability %"), 100.0)),
                  "jitter": int(_num(row.get("jitter"), 0)),
                  "offset": int(_num(row.get("offset"), 0))})
        for k in ("target", "group", "same_time_as"):
            v = _num(row.get(k))
            d[k] = v if v else None
        dur = _num(row.get("duration"))
        d["duration"] = None if dur is None else int(dur)
        if d["same_time_as"]:
            d.pop("year", None)
        else:
            d["year"] = int(_num(row.get("year"), 0))
        out.append({k: v for k, v in d.items() if v is not None})
    return out


def rows_to_lifestyle() -> dict | None:
    if tier_df is None:
        return base_raw.get("lifestyle")
    tiers = []
    for row in tier_df.to_dict("records"):
        if not isinstance(row.get("name"), str) or not row["name"].strip():
            continue
        monthly = {c: float(v) for c in ls0.categories
                   if (v := _num(row.get(c))) is not None}
        tiers.append({"name": row["name"].strip(), "take_home": float(_num(row.get("take_home"), 0.0)),
                      "monthly": monthly})
    tiers.sort(key=lambda t: t["take_home"])
    return {**base_raw["lifestyle"], "creep": round(creep_in / 100, 10),
            "withdrawal_rate": round(wr_in / 100, 10), "tiers": tiers}


what_if, what_if_error = None, None
try:
    raw = {**base_raw,
           "name": WHAT_IF,
           "portfolio": {**base_raw["portfolio"], "starting_balance": float(start_balance)},
           "income": rows_to_streams(inc_df),
           "expenses": rows_to_streams(exp_df),
           "events": rows_to_events(ev_df)}
    if rows_to_lifestyle() is not None:
        raw["lifestyle"] = rows_to_lifestyle()
    candidate = from_dict(raw)
    if replace(candidate, name=base.name, source=None) != replace(base, source=None):
        what_if = candidate
except (ScenarioError, ValueError, TypeError, KeyError) as exc:
    what_if_error = str(exc)

with tab_edit:
    if what_if_error:
        st.error(f"What-if is not valid yet: {what_if_error}")
    elif what_if is None:
        st.info("No changes from the reference yet.")
    else:
        override = {"extends": base_file.name, "name": "",
                    **diff(base, replace(what_if, name=base.name))}
        kind = "career" if profile_file else "scenario"
        st.subheader(f"Save as a {kind} file")
        new_name = st.text_input("Name", value=f"{base_name} (what if)")
        override["name"] = new_name
        text = yaml.safe_dump(override, sort_keys=False, allow_unicode=True)
        st.code(text, language="yaml")
        fname = f"{_slug(new_name)}.yaml"
        cdl, csave = st.columns(2)
        cdl.download_button("Download YAML", text, file_name=fname, mime="text/yaml")
        host = (st.context.headers.get("Host") or "").split(":")[0]
        if host not in ("localhost", "127.0.0.1", "::1"):
            csave.caption("Saving to the app's folders only works when you run the app on your "
                          "own computer. On the hosted app, download the file and add it to the "
                          "repo (shared server files would be visible to everyone and lost on "
                          "redeploy).")
        elif save_dir is None:
            csave.caption("In Strategies mode, download the change and save it as a career or "
                          "strategy file yourself.")
        else:
            target_file = save_dir / fname
            overwrite = target_file.exists()
            if csave.button(f"{'Overwrite' if overwrite else 'Save to'} {save_dir.name}/{fname}",
                            type="secondary" if overwrite else "primary"):
                target_file.write_text(text, encoding="utf-8")
                st.success(f"Saved {target_file.name}. It now appears in the lists.")


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

active = [shared(base)] + [shared(scenarios[n][1]) for n in chosen]
if what_if is not None:
    active.append(shared(what_if))
results = [run(s) for s in active]
_domain = [r.scenario.name for r in results]
color_scale = (alt.Scale(domain=_domain, range=list(COLORS[: len(results)]))
               if len(results) <= len(COLORS) else alt.Scale(domain=_domain, scheme="tableau20"))


def ages(res):
    a0 = res.scenario.start_age or 0
    return np.arange(res.scenario.horizon_years + 1) + a0


x_title = "Age" if base.start_age is not None else "Year"
a0 = base.start_age or 0

with tab_results:
    st.markdown(
        f"**{paths:,} paths**, seed {int(seed)}, all options on the same random draws. "
        f"*P(fail)* = the portfolio drops below {money(floor)} at any point, even briefly "
        "(a cash crunch counts). All dollars are today's dollars."
    )

    if profile_file:
        st.subheader("At a glance")
        g_name = st.selectbox("Option", [r.scenario.name for r in results], key="glance")
        g = next(r for r in results if r.scenario.name == g_name)
        g_a0 = g.scenario.start_age or 0
        g65 = min(65 - g_a0, g.scenario.horizon_years)
        g_inc = income_milestones(g, (200_000,))[0]
        g_nw = net_worth_milestones(g, (1e6,))[0]
        g_h, g_d, g_s = hours_metrics(g), debt_metrics(g), strategy_metrics(g)

        def _ms(m):
            return ("never" if not m.p_ever else
                    f"{100 * m.p_ever:.0f}% · age {m.median_age:.0f}")

        c1, c2 = st.columns(2)
        c1.metric("Net worth at 65 (median)", money(float(np.median(g.net_worth[:, g65]))),
                  help="Portfolio minus loans owed, today's dollars.")
        c2.metric("$200k take-home reached", _ms(g_inc),
                  help="Chance, and median age among paths that get there.")
        c3, c4 = st.columns(2)
        c3.metric("$1M net worth reached", _ms(g_nw))
        if g_s is not None:
            c4.metric(f"{g_s['strategy']} kicks in",
                      f"{100 * g_s['p_hit']:.0f}% · age {g_s['median_age']:.0f}"
                      if g_s["median_age"] else "never")
        elif g_h is not None:
            c4.metric("Take-home per work hour", money(g_h.take_home_per_hour),
                      help=f"Lifetime median: {g_h.lifetime_hours:,.0f} work hours.")
        if g_d is not None:
            st.caption(f"Loans: {100 * g_d.p_borrow:.0f}% of paths borrow, median peak "
                       f"{money(g_d.peak_median)}, paid off around age {g_d.debt_free_age:.0f}.")

        st.subheader("Scorecard")
        st.caption("Net worth = portfolio minus loans still owed (student loans count). "
                   + (f"Life events: {', '.join(file_name(f) for f in life_files)}. "
                      if life_files else "")
                   + (f"Strategy: {file_name(strategy_file)}." if strategy_file else ""))
        c1, c2 = st.columns(2)
        at_age = c1.slider(f"Evaluate at {x_title.lower()}", a0 + 5,
                           a0 + base.horizon_years, min(a0 + base.horizon_years, 65))
        target = c2.number_input("Target net worth ($)", value=2_000_000.0, step=250_000.0)
        rows = []
        for res in results:
            m = career_metrics(res, at_age, target)
            s = summarize(res)
            dm, hm, sm_ = debt_metrics(res), hours_metrics(res), strategy_metrics(res)
            row = {
                "Option": m.name,
                f"Median at {at_age}": m.net_worth_at_age[50],
                f"P10 at {at_age}": m.net_worth_at_age[10],
                f"P90 at {at_age}": m.net_worth_at_age[90],
                f"Chance ≥ {money(target)}": m.p_reach,
                "Lifetime pay (median, pre-tax)": m.lifetime_income,
                "Lifetime tax (median)": m.lifetime_tax,
                "Peak year pay (median)": m.peak_income,
                "Peak loans (median, borrowers)": dm.peak_median if dm else 0.0,
                "Loans paid off at age": (f"{dm.debt_free_age:.0f} ({100 * dm.p_borrow:.0f}% borrow)"
                                          if dm and dm.debt_free_age else "no loans"),
                "Take-home per work hour": hm.take_home_per_hour if hm else np.nan,
                "Years below $0 net worth (avg)": m.years_in_debt,
                "P(fail)": s.p_fail,
            }
            if any(r.scenario.strategy and r.scenario.strategy.kind != "work" for r in results):
                row["Strategy kicks in"] = (
                    "n/a (works to 65)" if sm_ is None else
                    f"{100 * sm_['p_hit']:.0f}% · age {sm_['median_age']:.0f}"
                    if sm_["median_age"] else "never")
            rows.append(row)
        df = pd.DataFrame(rows)
        money_cols = [c for c in df.columns
                      if c.startswith(("Median", "P10", "P90", "Lifetime", "Peak", "Take-home"))]
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={
                **{c: st.column_config.NumberColumn(c, format="$%,.0f") for c in money_cols},
                f"Chance ≥ {money(target)}": st.column_config.ProgressColumn(
                    f"Chance ≥ {money(target)}", min_value=0, max_value=1, format="percent"),
                "P(fail)": st.column_config.NumberColumn("P(fail)", format="percent"),
                "Years below $0 net worth (avg)": st.column_config.NumberColumn(
                    "Years below $0 net worth (avg)", format="%.1f"),
            },
        )


        st.subheader("Net worth at checkpoints")
        checks = [a for a in (30, 35, 40, 45, 50, 60, 65)
                  if a0 <= a <= a0 + base.horizon_years]
        q_label = {25: "P25", 50: "Median", 75: "P75"}
        cp_rows = []
        for res in results:
            r_a0 = res.scenario.start_age or 0
            row = {"Option": res.scenario.name}
            for a in checks:
                p25, p50, p75 = np.percentile(res.net_worth[:, a - r_a0], [25, 50, 75])
                row[f"Age {a}"] = f"{money(p50)} ({money(p25)} to {money(p75)})"
            cp_rows.append(row)
        st.dataframe(pd.DataFrame(cp_rows), hide_index=True, width="stretch")
        st.caption("Median net worth, with the middle half of paths (P25 to P75) in "
                   "brackets. Long-training paths start behind and catch up later, so "
                   "compare the columns, not just the last one.")

        acct_rows = []
        for res in results:
            ac = accounts_at(res, at_age)
            if ac:
                acct_rows.append({"Option": res.scenario.name, "Cash": ac["cash"],
                                  "Taxable": ac["taxable"], "401(k) (pre-tax)": ac["traditional"],
                                  "Roth": ac["roth"], "HSA": ac["hsa"],
                                  "Net worth": ac["pre_tax"], "After tax": ac["after_tax"]})
        if acct_rows:
            st.subheader(f"Accounts at {at_age}")
            st.dataframe(pd.DataFrame(acct_rows), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="$%,.0f")
                                        for c in ("Cash", "Taxable", "401(k) (pre-tax)", "Roth",
                                                  "HSA", "Net worth", "After tax")})
            st.caption("Medians of each account (they don't add up exactly). After tax counts "
                       "the tax still owed: income tax on 401(k) money when it comes out "
                       "(profile's retirement_tax_rate) and capital gains on the taxable account. "
                       "Contributions follow IRS 2025 limits while you have a paycheck.")

        debt_res = [r for r in results if r.debt is not None and r.debt.any()]
        if debt_res:
            st.subheader("Loans owed by age")
            dd = pd.concat([pd.DataFrame({"x": ages(r), "scenario": r.scenario.name,
                                          "owed": r.debt.mean(axis=0)}) for r in debt_res])
            show_chart(
                alt.Chart(dd).mark_line(strokeWidth=2.5).encode(
                    x=alt.X("x:Q", title=x_title),
                    y=alt.Y("owed:Q", title="Owed ($, today's)", axis=alt.Axis(format="$,.2s")),
                    color=alt.Color("scenario:N", scale=color_scale, title=None),
                    tooltip=["scenario:N", alt.Tooltip("x:Q", title=x_title),
                             alt.Tooltip("owed:Q", format="$,.0f")],
                ).properties(height=220),
                width="stretch")
            st.caption("Averages include paths that never borrow (for example, physician paths "
                       "where admission fails), so the typical borrower owes more; see the peak "
                       "loans column.")
    else:
        cols = st.columns(len(results))
        for col, res in zip(cols, results):
            s = summarize(res)
            delta = None
            if res is not results[0]:
                d = compare(results[0], res)
                delta = f"{100 * d.delta:+.1f} pts vs {base_name}"
            col.metric(res.scenario.name, f"{100 * s.p_fail:.1f}%", delta, delta_color="inverse",
                       help=f"95% CI {100 * s.ci[0]:.1f}% to {100 * s.ci[1]:.1f}%")
            col.caption(f"Ends below floor {100 * s.p_end_below:.1f}% · median end "
                        f"{money(s.terminal[50])}")

    bands = pd.concat([
        pd.DataFrame({"x": ages(res), "scenario": res.scenario.name,
                      "p10": (sm := summarize(res)).bands[10], "p50": sm.bands[50],
                      "p90": sm.bands[90]})
        for res in results
    ])
    st.subheader("Net worth by age")
    st.caption("Median line, shaded P10 to P90. Net worth = portfolio minus loans owed.")
    show = st.multiselect("Bands to shade", [r.scenario.name for r in results],
                          default=[results[0].scenario.name] + ([WHAT_IF] if what_if else []))
    area = alt.Chart(bands[bands.scenario.isin(show)]).mark_area(opacity=0.15).encode(
        x=alt.X("x:Q", title=x_title),
        y=alt.Y("p10:Q", title="Net worth (today's $)", axis=alt.Axis(format="$,.2s")),
        y2="p90:Q", color=alt.Color("scenario:N", scale=color_scale, title=None),
    )
    line = alt.Chart(bands).mark_line(strokeWidth=2.5).encode(
        x="x:Q", y="p50:Q", color=alt.Color("scenario:N", scale=color_scale, title=None),
        tooltip=[alt.Tooltip("scenario:N"), alt.Tooltip("x:Q", title=x_title),
                 alt.Tooltip("p10:Q", format="$,.0f", title="P10"),
                 alt.Tooltip("p50:Q", format="$,.0f", title="Median"),
                 alt.Tooltip("p90:Q", format="$,.0f", title="P90")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [floor]})).mark_rule(color="gray").encode(y="y:Q")
    show_chart((area + line + zero).properties(height=420).interactive(bind_y=False),
                    width="stretch")

    if any(r.income is not None and r.income.any() for r in results):
        basis = st.radio("Income shown", ["Take-home (after tax)", "Gross (before tax)"],
                         horizontal=True)
        take = basis.startswith("Take")
        st.subheader(f"{'Take-home' if take else 'Gross'} income by age")
        series = {res.scenario.name: (res.take_home if take else res.income) for res in results}
        inc = pd.concat([
            pd.DataFrame({"x": ages(res)[:-1], "scenario": res.scenario.name,
                          "p10": np.percentile(series[res.scenario.name], 10, axis=0),
                          "p50": np.percentile(series[res.scenario.name], 50, axis=0),
                          "p90": np.percentile(series[res.scenario.name], 90, axis=0),
                          "mean": series[res.scenario.name].mean(axis=0)})
            for res in results
        ])
        inc_area = alt.Chart(inc[inc.scenario.isin(show)]).mark_area(opacity=0.15).encode(
            x=alt.X("x:Q", title=x_title),
            y=alt.Y("p10:Q", title="Income ($/yr, today's $)", axis=alt.Axis(format="$,.2s")),
            y2="p90:Q", color=alt.Color("scenario:N", scale=color_scale, title=None))
        inc_line = alt.Chart(inc).mark_line(strokeWidth=2.5).encode(
            x="x:Q", y="p50:Q", color=alt.Color("scenario:N", scale=color_scale, title=None),
            tooltip=["scenario:N", alt.Tooltip("x:Q", title=x_title),
                     alt.Tooltip("p50:Q", format="$,.0f", title="Median"),
                     alt.Tooltip("mean:Q", format="$,.0f", title="Mean"),
                     alt.Tooltip("p10:Q", format="$,.0f", title="P10"),
                     alt.Tooltip("p90:Q", format="$,.0f", title="P90")])
        layers = [inc_area, inc_line]
        if take:
            marks = pd.DataFrame({"y": list(INCOME_MILESTONES),
                                  "label": [f"{money(v)} take-home" for v in INCOME_MILESTONES]})
            layers.append(alt.Chart(marks).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q"))
            layers.append(alt.Chart(marks).mark_text(align="left", dx=4, dy=-6, color="gray",
                                                     fontSize=11).encode(
                y="y:Q", text="label:N", x=alt.value(0)))
        show_chart(alt.layer(*layers).properties(height=320), width="stretch")
        st.caption("Dashed lines mark the take-home milestones (see the Milestones tab). Mean (in "
                   "the tooltip) can sit far above the median for careers with rare big outcomes.")

    modes = {res.scenario.name: failure_modes(res) for res in results}
    st.subheader("Share of paths that have failed, by age")
    fail_kind = st.radio("Failure means", list(FAILURE_MODES), horizontal=True,
                         format_func=lambda k: {"crunch": "Cash crunch",
                                                "insolvent": "Negative net worth after 40",
                                                "retirement": "Runs out after 65"}[k],
                         help="\n\n".join(FAILURE_MODES.values()))
    fail = pd.concat([
        pd.DataFrame({"x": ages(res)[1:], "scenario": res.scenario.name,
                      "failed": modes[res.scenario.name][fail_kind]["cumulative"]})
        for res in results
    ])
    st.caption(FAILURE_MODES[fail_kind] + ".")
    show_chart(
        alt.Chart(fail).mark_line(strokeWidth=2.5).encode(
            x=alt.X("x:Q", title=x_title),
            y=alt.Y("failed:Q", title="Failed by now", axis=alt.Axis(format="%")),
            color=alt.Color("scenario:N", scale=color_scale, title=None),
            tooltip=["scenario:N", alt.Tooltip("x:Q", title=x_title),
                     alt.Tooltip("failed:Q", format=".1%")],
        ).properties(height=240),
        width="stretch",
    )

    table = []
    for res in results:
        s = summarize(res)
        row = {
            "Option": s.name, "Cash crunch (P(fail))": f"{100 * s.p_fail:.1f}%",
            "95% CI": f"{100 * s.ci[0]:.1f}% to {100 * s.ci[1]:.1f}%",
            "Ends below": f"{100 * s.p_end_below:.1f}%",
            "Median first failure": "-" if s.median_failure_year is None else
            (f"age {res.scenario.start_age + s.median_failure_year:.0f}"
             if res.scenario.start_age is not None else f"year {s.median_failure_year:.0f}"),
            "Negative net worth after 40": f"{100 * modes[s.name]['insolvent']['p']:.1f}%",
            "Runs out after 65": f"{100 * modes[s.name]['retirement']['p']:.1f}%",
            "P10 end": money(s.terminal[10]), "Median end": money(s.terminal[50]),
            "P90 end": money(s.terminal[90]),
        }
        if res is not results[0]:
            d = compare(results[0], res)
            row[f"vs {base_name} (paired 95% CI)"] = (
                f"{100 * d.delta:+.1f} pts [{100 * d.ci[0]:+.1f}, {100 * d.ci[1]:+.1f}]")
        table.append(row)
    st.subheader("Failure detail")
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("A failure rate describes your assumptions, not the future. With lifestyle tiers, "
               "spending adapts to income, so failure is rare; the Lifestyle tab shows the "
               "budget each option sustains. A cash crunch can be brief (a down payment, a job "
               "gap) and recoverable; running out after 65 is the outcome to avoid.")

    rw = [(res.scenario.name, runway_metrics(res)) for res in results]
    rw = [(n, m) for n, m in rw if m]
    if rw:
        st.subheader("Startup runway")
        st.dataframe(pd.DataFrame([{
            "Option": n,
            "Runway years": f"age {a0 + m['window'][0]} to {a0 + m['window'][1]}",
            "Cash at start (median)": m["cash_at_start"],
            "Peak cash (median)": m["peak_cash"],
            "Cash ran out": m["p_cash_out"],
            "Liquid assets ran out": m["p_broke"],
        } for n, m in rw]), hide_index=True, width="stretch", column_config={
            "Cash at start (median)": st.column_config.NumberColumn(format="$%,.0f"),
            "Peak cash (median)": st.column_config.NumberColumn(format="$%,.0f"),
            "Cash ran out": st.column_config.NumberColumn(format="percent"),
            "Liquid assets ran out": st.column_config.NumberColumn(format="percent"),
        })
        st.caption("Founder careers have a runway window. Add a runway life event (4 or 2 "
                   "years in cash) to hold it in cash; without one, the founding years are paid "
                   "by selling investments, so 'cash ran out' is expected and only 'liquid "
                   "assets ran out' is a real failure.")


# ----------------------------------------------------------------------------
# Milestones
# ----------------------------------------------------------------------------


def _cell(m) -> str:
    if not m.p_ever:
        return "never"
    p = "<1%" if m.p_ever < 0.005 else f"{100 * m.p_ever:.0f}%"
    return f"{p} · age {m.median_age:.0f}" if base.start_age is not None \
        else f"{p} · year {m.median_age:.0f}"


with tabs["Milestones"]:
    st.markdown("All amounts are **today's dollars** (inflation already removed), so $300k means "
                "$300k of today's buying power. Each cell: chance the milestone is ever reached, "
                "and the median age among the paths that reach it.")
    mc1, mc2 = st.columns(2)
    inc_levels = mc1.multiselect("Take-home milestones ($/yr)",
                                 [100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 500_000],
                                 default=list(INCOME_MILESTONES), format_func=money)
    default_wr = base.lifestyle.withdrawal_rate if base.lifestyle else 0.035
    wr = mc2.number_input("Perpetual withdrawal rate (%)", 1.0, 6.0, round(default_wr * 100, 2), 0.25,
                          help="Share of the portfolio you could spend every year, forever, in "
                               "today's dollars. 3-3.5% is a common conservative choice for a "
                               "plan with no end date; 4% is the classic 30-year rule.") / 100

    inc_ms = {r.scenario.name: income_milestones(r, sorted(inc_levels)) for r in results}
    if inc_levels:
        st.subheader("Take-home pay milestones")
        cols = [f"{money(m.level)} take-home (≈{money(m.gross_needed)} gross)"
                for m in next(iter(inc_ms.values()))]
        st.dataframe(pd.DataFrame([{"Option": n, **{c: _cell(m) for c, m in zip(cols, ms)}}
                                   for n, ms in inc_ms.items()]),
                     hide_index=True, width="stretch")
        st.caption("Gross equivalents use the scenario's tax table (single filer, California). "
                   "One-time windfalls such as an exit payout are not take-home pay; they count "
                   "toward net worth.")

    nw_ms = {r.scenario.name: net_worth_milestones(r, NET_WORTH_MILESTONES, wr) for r in results}
    st.subheader("Net worth milestones")
    cols = [f"{money(m.level)} (→ {money(m.dividend)}/yr forever)"
            for m in next(iter(nw_ms.values()))]
    st.dataframe(pd.DataFrame([{"Option": n, **{c: _cell(m) for c, m in zip(cols, ms)}}
                               for n, ms in nw_ms.items()]),
                 hide_index=True, width="stretch")
    st.caption(f"The bracket is the perpetual dividend: what that portfolio could pay you every "
               f"year at {100 * wr:.2f}% if you stopped working, before tax, in today's dollars.")

    st.subheader("Reached by each age")
    options = ([f"Take-home {money(v)}" for v in sorted(inc_levels)]
               + [f"Net worth {money(v)}" for v in NET_WORTH_MILESTONES])
    pick = st.selectbox("Milestone", options, index=min(len(inc_levels), len(options) - 1))
    rows = []
    for res in results:
        ms = inc_ms[res.scenario.name] if pick.startswith("Take") else nw_ms[res.scenario.name]
        m = next(m for m in ms if pick.endswith(money(m.level)))
        x = ages(res)[:-1] if pick.startswith("Take") else ages(res)[1:]
        rows.append(pd.DataFrame({"x": x, "scenario": res.scenario.name, "p": m.by_age}))
    show_chart(
        alt.Chart(pd.concat(rows)).mark_line(strokeWidth=2.5).encode(
            x=alt.X("x:Q", title=x_title),
            y=alt.Y("p:Q", title="Reached by now", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("scenario:N", scale=color_scale, title=None),
            tooltip=["scenario:N", alt.Tooltip("x:Q", title=x_title),
                     alt.Tooltip("p:Q", format=".0%", title="Reached")],
        ).properties(height=300),
        width="stretch",
    )


# ----------------------------------------------------------------------------
# Lifestyle: which budget each option sustains
# ----------------------------------------------------------------------------

with tabs["Lifestyle"]:
    ls_results = [r for r in results if r.tier is not None]
    if not ls_results:
        st.info("This scenario has no lifestyle tiers. Add a `lifestyle:` section to its file "
                "(see profiles/my_profile.yaml).")
    else:
        ls = base.lifestyle or ls_results[0].scenario.lifestyle
        st.markdown(
            f"Each year, **affordable income** = average take-home over the last "
            f"{ls.smoothing_years} years + {100 * ls.withdrawal_rate:.1f}% of the portfolio. "
            f"It picks the highest tier you have reached; between tiers you also spend "
            f"{100 * ls.creep:.0f}% of every extra dollar (lifestyle creep). Edit tiers on the "
            "What if tab or in the profile file.")
        cats = list(ls.categories)
        budget = pd.DataFrame(
            {"Category": cats + ["Total / month", "Total / year"]} |
            {t.name + f" (≥{money(t.take_home)})": [dict(t.monthly).get(c, np.nan) for c in cats]
             + [sum(a for _, a in t.monthly), 12 * sum(a for _, a in t.monthly)] for t in ls.tiers})
        st.subheader("Tier budgets")
        st.dataframe(budget, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="$%,.0f")
                                    for c in budget.columns if c != "Category"})

        pick_ls = st.selectbox("Option", [r.scenario.name for r in ls_results], key="ls_pick")
        res = next(r for r in ls_results if r.scenario.name == pick_ls)
        shares = tier_shares(res)
        tiers_df = pd.concat([pd.DataFrame({"x": ages(res)[:-1], "tier": n, "share": v})
                              for n, v in shares.items()])
        st.subheader("Lifestyle tier by age")
        show_chart(
            alt.Chart(tiers_df).mark_area().encode(
                x=alt.X("x:Q", title=x_title),
                y=alt.Y("share:Q", stack="normalize", title="Share of paths",
                        axis=alt.Axis(format="%")),
                color=alt.Color("tier:N", sort=list(shares), title=None,
                                scale=alt.Scale(domain=list(shares),
                                                range=["#c9ced6", "#9cc3e6", "#2a6fdb",
                                                       "#e07b39", "#2e9e6a", "#a94fc9"][: len(shares)])),
                order=alt.Order("tier_order:Q"),
                tooltip=["tier:N", alt.Tooltip("x:Q", title=x_title),
                         alt.Tooltip("share:Q", format=".0%")],
            ).transform_calculate(tier_order=f"indexof({list(shares)}, datum.tier)")
            .properties(height=300),
            width="stretch",
        )

        st.subheader("Yearly spending by category")
        spend = res.flow_means["spending"]
        sp = pd.concat([pd.DataFrame({"x": ages(res)[:-1], "category": c, "amount": v})
                        for c, v in spend.items() if v.any()])
        show_chart(
            alt.Chart(sp).mark_area().encode(
                x=alt.X("x:Q", title=x_title),
                y=alt.Y("amount:Q", stack=True, title="$/yr (today's $)",
                        axis=alt.Axis(format="$,.2s")),
                color=alt.Color("category:N", title=None),
                tooltip=["category:N", alt.Tooltip("x:Q", title=x_title),
                         alt.Tooltip("amount:Q", format="$,.0f")],
            ).properties(height=300),
            width="stretch",
        )

        rows = []
        for r in ls_results:
            sh = tier_shares(r)
            a = r.scenario.start_age or 0
            row = {"Option": r.scenario.name}
            for age in (35, 45, 55, 65, 75):
                t = age - a
                if 0 <= t < r.scenario.horizon_years:
                    best = max(sh, key=lambda n: sh[n][t])
                    row[f"Most common at {age}"] = f"{best} ({100 * sh[best][t]:.0f}%)"
            rows.append(row)
        st.subheader("Most common tier by age")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------
# Hours: how much of your life each option takes
# ----------------------------------------------------------------------------

with tabs["Hours"]:
    hour_res = [r for r in results if r.hours is not None]
    if not hour_res:
        st.info("These options have no work-hours schedule. Add a `time:` section "
                "(see careers/rd_engineer.yaml).")
    else:
        st.markdown("Hours per week at work, by age. Each career sets its hours for each "
                    "phase (school, residency, associate, founder...) from survey data, listed "
                    "under *Where the hours come from* below. Every career then eases about 5% at "
                    "50 and another 5% at 58, and swings about 5% year to year. Lifetime totals "
                    f"assume {WEEKS_PER_YEAR} working weeks a year; FIRE and coast strategies "
                    "cut hours when they kick in.")
        hrs = pd.concat([
            pd.DataFrame({"x": ages(r)[:-1], "scenario": r.scenario.name,
                          "p10": np.percentile(r.hours, 10, axis=0),
                          "p50": np.percentile(r.hours, 50, axis=0),
                          "p90": np.percentile(r.hours, 90, axis=0),
                          "mean": r.hours.mean(axis=0)})
            for r in hour_res])
        st.subheader("Hours per week by age")
        show_h = st.multiselect("Bands to shade", [r.scenario.name for r in hour_res],
                                default=[hour_res[0].scenario.name], key="hours_bands")
        band = alt.Chart(hrs[hrs.scenario.isin(show_h)]).mark_area(opacity=0.15).encode(
            x=alt.X("x:Q", title=x_title), y=alt.Y("p10:Q", title="Hours / week"), y2="p90:Q",
            color=alt.Color("scenario:N", scale=color_scale, title=None))
        line = alt.Chart(hrs).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
            x="x:Q", y="p50:Q", color=alt.Color("scenario:N", scale=color_scale, title=None),
            tooltip=["scenario:N", alt.Tooltip("x:Q", title=x_title),
                     alt.Tooltip("p50:Q", format=".0f", title="Median h/wk"),
                     alt.Tooltip("mean:Q", format=".1f", title="Mean h/wk")])
        ref = alt.Chart(pd.DataFrame({"y": [40]})).mark_rule(strokeDash=[4, 4], color="gray"
                                                              ).encode(y="y:Q")
        show_chart(alt.layer(band, line, ref).properties(height=320), width="stretch")
        st.caption("Dashed line: a 40-hour week. Options with uncertain paths (admission, "
                   "tenure, partnership) mix outcomes in the median; the shaded band shows the spread.")

        rows = []
        for r in hour_res:
            hm = hours_metrics(r)
            row = {"Option": hm.name,
                   "Lifetime work hours (median)": hm.lifetime_hours,
                   "Years working (median)": hm.years_worked,
                   **{f"Avg h/wk in {k}": v for k, v in hm.by_decade.items() if v},
                   "Lifetime take-home (median)": hm.lifetime_take_home,
                   "Take-home per work hour": hm.take_home_per_hour}
            rows.append(row)
        df = pd.DataFrame(rows)
        st.subheader("Time cost of each option")
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Lifetime work hours (median)": st.column_config.NumberColumn(format="%,.0f"),
            "Years working (median)": st.column_config.NumberColumn(format="%.0f"),
            "Lifetime take-home (median)": st.column_config.NumberColumn(format="$%,.0f"),
            "Take-home per work hour": st.column_config.NumberColumn(format="$%,.0f"),
            **{c: st.column_config.NumberColumn(format="%.0f")
               for c in df.columns if c.startswith("Avg h/wk")},
        })
        st.caption("Take-home per work hour = lifetime take-home pay ÷ lifetime work hours "
                   "(Ph.D., school and training years included).")

        st.subheader("Free time")
        commute = st.slider("Commute (hours a week, while working)", 0.0, 15.0, 4.5, 0.5,
                            help="US average: about 27 minutes each way (Census ACS), ~4.5 "
                                 "hours a week.")
        ft_rows = []
        for r in hour_res:
            ft = free_time(r, commute)
            ft_rows.append({"Option": ft.name, "Lifetime free hours": ft.lifetime_free,
                            "Free hours before 65": ft.free_before_65,
                            "Years not working after the Ph.D.": ft.retired_years,
                            "Share of waking hours free": ft.free_share})
        st.dataframe(pd.DataFrame(ft_rows), hide_index=True, width="stretch", column_config={
            "Lifetime free hours": st.column_config.NumberColumn(format="%,.0f"),
            "Free hours before 65": st.column_config.NumberColumn(format="%,.0f"),
            "Years not working after the Ph.D.": st.column_config.NumberColumn(format="%.0f"),
            "Share of waking hours free": st.column_config.NumberColumn(format="percent"),
        })
        st.caption("Free hours = 16 waking hours a day minus work and commuting, with 4 weeks a "
                   "year off, from now to the end of the plan. Chores, sleep beyond 8 hours and "
                   "caregiving are not taken out, so compare the options rather than the totals.")

        st.subheader("Where the hours come from")
        hour_files = ([career_file] if mode == "Strategies" else
                      [scenarios[r.scenario.name][0] for r in hour_res])
        prof_lines = [s for s in career_info(profile_file)["sources"] if s.startswith("Hours")]
        if prof_lines:
            with st.expander("Profile: Ph.D. hours and the age taper"):
                st.markdown("\n".join(f"- {s.removeprefix('Hours: ')}" for s in prof_lines))
        for f in dict.fromkeys(hour_files):
            info = career_info(f)
            notes = [a for a in info["assumptions"] if a.startswith("Hours")]
            srcs = [s for s in info["sources"] if s.startswith("Hours")]
            if not (notes or srcs):
                continue
            with st.expander(info["name"]):
                st.markdown("\n".join(f"- {a.removeprefix('Hours: ')}" for a in notes))
                if srcs:
                    st.markdown("**Sources**\n\n" + "\n".join(
                        f"- {s.removeprefix('Hours: ')}" for s in srcs))


# ----------------------------------------------------------------------------
# Decide: weighted scores and Pareto view
# ----------------------------------------------------------------------------

if profile_file:
    with tabs["Decide"]:
        st.markdown(
            "Money is one part of the choice. Set how much each criterion matters to you; "
            "every option gets a 0-100 score per criterion (0 = worst option here, 100 = best), "
            "then a weighted total. Autonomy, satisfaction and burnout come from surveys listed "
            "in each career file (Assumptions tab); several are marked CHECK.")
        with st.expander("Weights (0 = ignore, 10 = matters most)", expanded=True):
            wcols = st.columns(2)
            weights = {}
            for i, c in enumerate(CRITERIA):
                weights[c.key] = wcols[i % 2].slider(
                    c.label, 0, 10, DEFAULT_WEIGHTS[c.key], key=f"w_{c.key}", help=c.help)
        scored = score(results, weights)
        rows = []
        for s_ in scored:
            v = s_.values
            rows.append({
                "Option": s_.name, "Score": s_.total, "Financial": s_.financial, "Life": s_.life,
                "Net worth 65": v["wealth"], "P10 at 65": v["downside"],
                "FI age": v["fi_age"], "Low-pay years": v["training"],
                "Lifetime hours": v["hours"], "Autonomy": v["autonomy"],
                "Satisfied": v["satisfaction"], "Burnout": v["burnout"],
            })
        sdf = pd.DataFrame(rows)
        st.subheader("Ranking")
        st.dataframe(sdf, hide_index=True, width="stretch", column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                     format="%.0f"),
            "Financial": st.column_config.NumberColumn(format="%.0f"),
            "Life": st.column_config.NumberColumn(format="%.0f"),
            "Net worth 65": st.column_config.NumberColumn(format="$%,.0f"),
            "P10 at 65": st.column_config.NumberColumn(format="$%,.0f"),
            "FI age": st.column_config.NumberColumn(format="%.0f"),
            "Low-pay years": st.column_config.NumberColumn(format="%.0f"),
            "Lifetime hours": st.column_config.NumberColumn(format="%,.0f"),
            "Autonomy": st.column_config.NumberColumn(format="%.0f"),
            "Satisfied": st.column_config.NumberColumn(format="percent"),
            "Burnout": st.column_config.NumberColumn(format="percent"),
        })
        end_age = a0 + base.horizon_years
        st.caption(f"FI age: first age the portfolio reaches 25x that year's spending (an age of "
                   f"{end_age} means the median path never does). Financial = weighted score of "
                   "the money criteria only; Life = the rest.")

        st.subheader("Pareto view")
        axis_opts = ["Financial score", "Life score", *[c.label for c in CRITERIA]]
        pc1, pc2 = st.columns(2)
        x_lab = pc1.selectbox("Across", axis_opts, index=0, key="pareto_x")
        y_lab = pc2.selectbox("Up", axis_opts, index=1, key="pareto_y")
        by_label = {c.label: c for c in CRITERIA}

        def axis_value(s_, lab):
            if lab == "Financial score":
                return s_.financial
            if lab == "Life score":
                return s_.life
            return s_.normalized[by_label[lab].key] * 100

        pts = [(axis_value(s_, x_lab), axis_value(s_, y_lab)) for s_ in scored]
        front = pareto(pts)
        pdf = pd.DataFrame({
            "Option": [s_.name for s_ in scored],
            "x": [p[0] for p in pts], "y": [p[1] for p in pts],
            "Frontier": ["On the frontier" if f else "Beaten on both" for f in front],
            "label": [s_.name if len(s_.name) <= 26 else s_.name[:25] + "…" for s_ in scored]})
        enc_x = alt.X("x:Q", title=f"{x_lab} (right is better)", scale=alt.Scale(domain=[-5, 105]))
        enc_y = alt.Y("y:Q", title=f"{y_lab} (up is better)", scale=alt.Scale(domain=[-5, 105]))
        fr = pdf[pdf.Frontier == "On the frontier"].sort_values("x")
        fline = alt.Chart(fr).mark_line(strokeDash=[5, 4], color="gray").encode(x=enc_x, y=enc_y)
        dots = alt.Chart(pdf).mark_circle(size=140, opacity=0.9).encode(
            x=enc_x, y=enc_y,
            color=alt.Color("Frontier:N", title=None,
                            scale=alt.Scale(domain=["On the frontier", "Beaten on both"],
                                            range=["#2a6fdb", "#b0b7c3"])),
            tooltip=["Option:N", alt.Tooltip("x:Q", format=".0f", title=x_lab),
                     alt.Tooltip("y:Q", format=".0f", title=y_lab)])
        labels = alt.Chart(pdf).mark_text(align="left", dx=8, dy=-6, fontSize=11).encode(
            x=enc_x, y=enc_y, text="label:N")
        show_chart(alt.layer(fline, dots, labels).properties(height=380),
                   width="stretch")
        st.caption("Each option on the frontier is the best available for some trade-off between "
                   "the two axes: nothing else beats it on both. An option beaten on both is "
                   "worth picking only for reasons outside these scores.")
        st.markdown("**Frontier:** " + ", ".join(s_.name for s_, f in zip(scored, front) if f))

        st.divider()
        st.subheader("Decision point")
        st.markdown("Keep an option open: follow path A, and at a chosen age switch to path B on "
                    "the lives where you can afford to. Both paths run on the same random draws, "
                    "so each life's luck before and after the choice is the same.")
        all_names = list(scenarios)
        dp1, dp2 = st.columns(2)
        ia = next((i for i, n in enumerate(all_names) if scenarios[n][0].stem == "rd_engineer"), 0)
        ib = next((i for i, n in enumerate(all_names)
                   if scenarios[n][0].stem == "save_then_md_ep"), min(1, len(all_names) - 1))
        dp_a = dp1.selectbox("Start on (A)", all_names, index=ia, key="dp_a")
        dp_b = dp2.selectbox("Maybe switch to (B)", all_names, index=ib, key="dp_b",
                             help="Pick a B whose changes start after the decision age, e.g. "
                                  "'Industry first, then MD / EP' with a decision at 29.")
        dp3, dp4, dp5 = st.columns(3)
        dp_age = dp3.number_input("Decide at age", a0 + 1, a0 + base.horizon_years - 1,
                                  min(a0 + 7, a0 + base.horizon_years - 1), key="dp_age")
        dp_rule = dp4.selectbox("Switch if", list(DP_RULES), format_func=DP_RULES.get,
                                key="dp_rule")
        dp_th = dp5.number_input("Threshold ($)", 0.0, 5e6, 250_000.0, 25_000.0, key="dp_th",
                                 disabled=dp_rule == "always")
        if st.button("Run the decision point", type="primary"):
            with st.spinner("Simulating A, B and the switch..."):
                dp = dp_decide(shared(scenarios[dp_a][1]), shared(scenarios[dp_b][1]),
                               int(dp_age - a0), dp_rule, float(dp_th))
            st.session_state["dp_result"] = dp
        dp = st.session_state.get("dp_result")
        if dp is not None:
            col = int(np.clip(65 - a0, 0, base.horizon_years))
            st.metric("Paths that switch", f"{100 * dp.p_switch:.0f}%")
            rows = []
            for lab, r_ in (("Stay on A", dp.a), ("Switch to B regardless", dp.b),
                            ("Decide at the point", dp.combined)):
                nw = r_.net_worth[:, col]
                rows.append({"Plan": lab, "Median at 65": np.median(nw),
                             "P10 at 65": np.percentile(nw, 10), "P90 at 65": np.percentile(nw, 90),
                             "Mean at 65": nw.mean(), "Cash crunch": r_.p_fail})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                **{c: st.column_config.NumberColumn(format="$%,.0f")
                   for c in ("Median at 65", "P10 at 65", "P90 at 65", "Mean at 65")},
                "Cash crunch": st.column_config.NumberColumn(format="percent")})
            sw = dp.switch
            if 0 < sw.mean() < 1:
                gain = dp.combined.net_worth[:, col] - dp.a.net_worth[:, col]
                st.caption(f"On the lives that switched, net worth at 65 vs staying on A: median "
                           f"{money(float(np.median(gain[sw])))}. {dp.name}.")
            else:
                st.caption(dp.name + ".")


# ----------------------------------------------------------------------------
# Cash flow Sankey
# ----------------------------------------------------------------------------

with tabs["Cash flow"]:
    import plotly.graph_objects as go

    sc1, sc2 = st.columns(2)
    pick_cf = sc1.selectbox("Option", [r.scenario.name for r in results], key="cf_pick")
    res = next(r for r in results if r.scenario.name == pick_cf)
    a = res.scenario.start_age or 0
    age_cf = sc2.slider(f"{x_title}", a, a + res.scenario.horizon_years - 1, min(a + 10, a + 35),
                        key="cf_age")
    t = age_cf - a
    fm = res.flow_means

    detail = st.radio("Detail", ["Summary", "Full"], horizontal=True, key="cf_detail",
                      help="Summary groups taxes and spending so the chart reads on a phone.")
    summary = detail == "Summary"

    def val(d, k):
        return float(d[k][t])

    ESSENTIALS = {"Housing", "Food", "Transport", "Health & insurance", "Utilities & phone",
                  "Healthcare in retirement"}
    tax_names = set(fm["tax"])

    def group(name: str) -> str:
        if not summary:
            return name
        if name in tax_names:
            return "Taxes"
        if name in ESSENTIALS:
            return "Essentials"
        if name.startswith("Child"):
            return "Kids"
        if name.startswith("Partner"):
            return "Partner"
        if name in ("One-time purchases", "Loan payments", "Saved", "From savings",
                    "Total income", "Take-home", "Windfalls (after tax)", "Loans received"):
            return name
        return "Lifestyle"

    flows_: dict[tuple[str, str], float] = {}

    def link(src: str, dst: str, v: float) -> None:
        if v > 1:
            key = (group(src) if src not in incomes else src, group(dst))
            flows_[key] = flows_.get(key, 0.0) + v

    gross, take = "Total income", "Take-home"
    incomes = {k: val(fm["income"], k) for k in fm["income"]}
    spendable = {s.name: s.spendable for s in res.scenario.income}
    for k, v in incomes.items():
        link(k, gross, v)
    tax_total = sum(val(fm["tax"], k) for k in fm["tax"])
    for k in fm["tax"]:
        link(gross, k, val(fm["tax"], k))
    nonspend = sum(v for k, v in incomes.items() if not spendable.get(k, True))
    link(gross, take, sum(incomes.values()) - tax_total - nonspend)
    other = {k: val(fm["other"], k) for k in fm["other"]}
    for k in ("Windfalls (after tax)", "Loans received"):
        link(k, take, other[k])
    spend_total = 0.0
    for k in fm["spending"]:
        v = val(fm["spending"], k)
        link(take, k, v)
        spend_total += max(v, 0.0)
    for k in ("One-time purchases", "Loan payments", "Home costs"):
        link(take, k, other.get(k, 0.0))
    saved = float(fm["saved"][t]) - nonspend
    if saved > 0:
        link(take, "Saved", saved)
    else:
        link("From savings", take, -saved)
    link(gross, "Saved", nonspend)

    nodes: list[str] = []
    for s_, d_ in flows_:
        for n in (s_, d_):
            if n not in nodes:
                nodes.append(n)
    idx = {n: i for i, n in enumerate(nodes)}
    total_of = {n: max(sum(v for (s_, d_), v in flows_.items() if d_ == n),
                       sum(v for (s_, d_), v in flows_.items() if s_ == n)) for n in nodes}
    palette = {"Saved": "#2e9e6a", "From savings": "#c9474f", take: "#2a6fdb", gross: "#6b7a8f",
               "Taxes": "#e07b39"}
    colors = [palette.get(n, "#e07b39" if n in tax_names else "#9cc3e6") for n in nodes]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[f"{n} {money(total_of[n])}" for n in nodes],
                  color=colors, pad=18 if summary else 12, thickness=16),
        link=dict(source=[idx[s_] for s_, _ in flows_], target=[idx[d_] for _, d_ in flows_],
                  value=list(flows_.values()),
                  hovertemplate="%{source.label} → %{target.label}: $%{value:,.0f}<extra></extra>"),
    ))
    fig.update_layout(height=420 if summary else 560, margin=dict(l=4, r=4, t=4, b=4),
                      font=dict(size=12))
    st.subheader(f"Where a year's money goes at {age_cf}")
    st.caption(f"{pick_cf}, average across paths, today's dollars.")
    st.plotly_chart(fig, width="stretch")

    income_total = sum(incomes.values())
    if income_total > 0:
        kept = max(saved, 0) + nonspend
        st.dataframe(pd.DataFrame([
            {"": "Total income", "$/yr": income_total, "Share": 1.0},
            {"": "Taxes", "$/yr": tax_total, "Share": tax_total / income_total},
            {"": "Spending", "$/yr": spend_total, "Share": spend_total / income_total},
            {"": "Saved (incl. employer match)", "$/yr": kept, "Share": kept / income_total},
        ]), hide_index=True, width="stretch", column_config={
            "$/yr": st.column_config.NumberColumn(format="$%,.0f"),
            "Share": st.column_config.NumberColumn(format="percent")})
        st.caption("Averages across paths can mix outcomes, such as admitted and not-admitted "
                   "paths for the physician.")


# ----------------------------------------------------------------------------
# Career assumptions and sources
# ----------------------------------------------------------------------------

def show_info(f: Path, expanded: bool) -> None:
    info = career_info(f)
    with st.expander(f"{info['name']}  ({f.parent.name}/{f.name})", expanded=expanded):
        if info["description"]:
            st.write(info["description"])
        if info["assumptions"]:
            st.markdown("**Assumptions**")
            st.markdown("\n".join(
                f"- :orange-badge[Low confidence] {a.replace('(CHECK)', '').replace('CHECK: ', '').replace(' CHECK', '').replace('CHECK', '')}"
                if "CHECK" in a else f"- {a}" for a in info["assumptions"]))
        if info["sources"]:
            st.markdown("**Sources**")
            st.markdown("\n".join(f"- {s}" for s in info["sources"]))


if profile_file:
    with tabs["Assumptions"]:
        st.markdown("Every number comes from the files below. Edit the YAML (or use the What if "
                    "tab) to change an assumption. Items marked CHECK are guesses rather than "
                    "sourced figures.")
        if mode == "Strategies":
            st.subheader("Career")
            show_info(career_file, True)
        for n in [base_name, *chosen]:
            show_info(scenarios[n][0], n == base_name)
        if life_files or strategy_file:
            st.subheader("Add-ons")
            for f in [*life_files, *([strategy_file] if strategy_file else [])]:
                show_info(f, False)


# ----------------------------------------------------------------------------
# Sensitivity
# ----------------------------------------------------------------------------

with tab_sens:
    target_name = st.selectbox("Option", [r.scenario.name for r in results],
                               index=len(results) - 1 if what_if else 0)
    target_s = next(r.scenario for r in results if r.scenario.name == target_name)
    c1, c2, c3, c4 = st.columns(4)
    money_step = c1.number_input("Dollar nudge (%)", 1.0, 50.0, 10.0, 1.0) / 100
    rate_step = c2.number_input("Return / inflation nudge (pts)", 0.1, 5.0, 1.0, 0.1) / 100
    year_step = int(c3.number_input("Timing nudge (yrs)", 0, 10, 2, 1))
    top = int(c4.number_input("Inputs shown", 3, 40, 15, 1))
    steps = Steps(money=money_step, rate=rate_step, years=year_step)

    key = ("sensitivity", target_s, steps)
    if key not in _cache():
        n_runs = 2 * len(knobs(target_s, steps))
        st.caption(f"Needs {n_runs} extra simulations of {target_s.paths:,} paths.")
        go = st.button("Run sensitivity", type="primary")
    else:
        go = True
    if go:
        with st.spinner("Nudging each input down and up..."):
            sa = run_sensitivity(target_s, steps)
        rows = sa.rows[:top]
        base_p = sa.base.p_fail
        order = [r.knob.label for r in rows]
        df = pd.DataFrame(
            [{"input": r.knob.label, "side": "input lower", "value": r.knob.low_desc,
              "p": r.p_low, "base": base_p} for r in rows]
            + [{"input": r.knob.label, "side": "input higher", "value": r.knob.high_desc,
                "p": r.p_high, "base": base_p} for r in rows]
        )
        chart = alt.Chart(df).mark_bar(height=14).encode(
            y=alt.Y("input:N", sort=order, title=None),
            x=alt.X("base:Q", title="Probability of failure", axis=alt.Axis(format="%")),
            x2="p:Q",
            color=alt.Color("side:N", scale=alt.Scale(domain=["input lower", "input higher"],
                                                      range=[COLORS[0], COLORS[1]]), title=None),
            tooltip=["input:N", "side:N", "value:N", alt.Tooltip("p:Q", format=".1%", title="P(fail)")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [base_p]})).mark_rule(color="black").encode(x="x:Q")
        st.subheader(f"What moves the failure rate (base {100 * base_p:.1f}%)")
        show_chart((chart + rule).properties(height=26 * len(rows) + 40),
                        width="stretch")
        st.dataframe(pd.DataFrame([{
            "Input": r.knob.label, "Low": r.knob.low_desc, "P(fail) low": f"{100 * r.p_low:.1f}%",
            "High": r.knob.high_desc, "P(fail) high": f"{100 * r.p_high:.1f}%",
            "Swing (pts)": round(100 * r.swing, 1),
        } for r in sa.rows]), hide_index=True, width="stretch")
        st.caption("Compare swings within one kind of nudge: a 10% salary change and a "
                   "1-point return change are different-sized questions.")


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

with tab_sens:
    st.divider()
    st.subheader("Break-even finder")
    st.markdown("What would have to be true for one option to match another? Pick an input of "
                "option A and the finder searches for the value at which A's yardstick equals "
                "B's, re-running A on the same random draws.")
    by_name = {r.scenario.name: r.scenario for r in results}
    opt_names = list(by_name)
    if len(opt_names) < 2:
        st.info("Compare at least two options to use the break-even finder.")
    else:
        default_a = next((i for i, n in enumerate(opt_names) if "electrophysiologist" in n), 0)
        b1, b2 = st.columns(2)
        a_name = b1.selectbox("Option A (the one to adjust)", opt_names, index=default_a,
                              key="be_a")
        b_choices = [n for n in opt_names if n != a_name]
        b_name = b2.selectbox("Option B (the benchmark)", b_choices, key="be_b")
        lv = be_levers(by_name[a_name])
        default_l = next((i for i, x in enumerate(lv) if "attending" in x.label.lower()
                          or "acquired" in x.label.lower()), 0)
        lever = st.selectbox("Input of option A to solve for", lv, index=default_l,
                             format_func=lambda x: f"{x.label}: now {money(x.value)}", key="be_l")
        b3, b4 = st.columns(2)
        metric = b3.selectbox("Yardstick", list(BE_METRICS), index=0, key="be_m",
                              help="Median: the typical path. Mean: includes rare big outcomes. "
                                   "P10: a bad-luck path.")
        be_age = b4.slider("At age", a0 + 10, a0 + base.horizon_years, min(65, a0 + base.horizon_years),
                           key="be_age")
        if st.button("Find the break-even", type="primary"):
            with st.spinner("Searching (about 20 runs at 2,000 paths)..."):
                be = be_solve(by_name[a_name], by_name[b_name], lever, metric, be_age)
            st.session_state["be_result"] = (a_name, b_name, be)
        saved = st.session_state.get("be_result")
        if saved and saved[0] == a_name and saved[1] == b_name and saved[2].lever == lever.label:
            _, _, be = saved
            c1, c2, c3 = st.columns(3)
            c1.metric(f"B: {metric.lower()}", money(be.target))
            c2.metric(f"A as written", money(be.base_metric))
            c3.metric("Break-even input", "none in range" if be.value is None else money(be.value),
                      None if be.value is None else f"{100 * (be.value / be.base_value - 1):+.0f}% vs now")
            st.caption(be.note + (f" This input only applies on {100 * be.reach:.0f}% of option "
                                  "A's paths, so the median may not move much however large it "
                                  "gets; try the mean." if be.reach < 0.5 else ""))
            cdf = pd.DataFrame(be.curve, columns=["input", "metric"])
            line = alt.Chart(cdf).mark_line(point=True).encode(
                x=alt.X("input:Q", title=lever.label, axis=alt.Axis(format="$,.2s")),
                y=alt.Y("metric:Q", title=f"A: {metric.lower()} at {be_age}",
                        axis=alt.Axis(format="$,.2s")))
            ref = alt.Chart(pd.DataFrame({"y": [be.target]})).mark_rule(
                strokeDash=[5, 4], color="gray").encode(y="y:Q")
            show_chart(alt.layer(line, ref).properties(height=280), width="stretch")
            st.caption("Dashed line: option B. Each dot is one re-run of option A.")


with tab_files:
    if profile_file:
        with st.expander(f"Profile ({profile_file.name})", expanded=True):
            st.code(profile_file.read_text(encoding="utf-8"), language="yaml")
    for n in [base_name, *chosen]:
        f, _ = scenarios[n]
        with st.expander(f"{n}  ({f.name})", expanded=not profile_file and n == base_name):
            st.code(f.read_text(encoding="utf-8"), language="yaml")
