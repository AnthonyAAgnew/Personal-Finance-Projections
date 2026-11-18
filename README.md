# STATS Projections

A small Monte Carlo life planner. It asks one question: across thousands of
possible futures, how often does my money run out? See
`Monte Carlo Life Planner Design Philosophy.md` for the reasoning behind it.

## Setup

```powershell
python -m pip install --user numpy pyyaml matplotlib pytest streamlit
```

## App

```powershell
python -m streamlit run app.py
```

- **Careers mode (default):** your profile (`profiles/my_profile.yaml`) plus
  each career path in `careers/`. All careers run on the same random draws.
  You get:
  - a **scorecard** with net worth at a chosen age, the chance of reaching a
    target, lifetime pay and taxes, peak pay, and years in debt;
  - portfolio and pre-tax income bands;
  - a **Career assumptions** tab listing every sourced number and every guess
    (marked CHECK).
- **Life scenarios mode:** the standalone files in `scenarios/`.
- **What if:** edit a copy of the reference, then save the change as a small
  override file that `extends` it.
- **Sensitivity:** a tornado chart for any option. Sidebar sliders change
  the market assumptions for every option at once.

## Careers

```powershell
python -m stats_projections --profile profiles/my_profile.yaml "careers/*.yaml" --chart outputs/careers.png
```

`profiles/my_profile.yaml` describes you today: age, savings, Ph.D.
stipend, living costs, Social Security, and retirement age. Each file in
`careers/` layers a path on top of it:

| File | Path | Key data |
| --- | --- | --- |
| `rd_engineer.yaml` | Industry med device R&D | BLS biomedical engineers, Levels.fyi Medtronic levels |
| `regulator.yaml` | FDA CDRH reviewer | GS-13/14/15 and Title 21 pay, FERS pension |
| `device_sales.yaml` | Clinical specialist, then territory rep | MedReps survey coverage, commission swings |
| `doctor.yaml` | MD after the Ph.D. | AAMC cost of attendance and debt, AAMC residency pay, Medscape 2026 |
| `entrepreneur.yaml` | Med device founder | Kruze founder pay, VC exit rates, PitchBook and PMCF exit data |

Each career file lists its **sources** and **assumptions**. Numbers marked
CHECK are guesses: replace them with your own. To add a career, copy a file
and edit it. Taxable income is gross pay, taxed with
`data/taxes_ca_single.yaml`, a progressive CA + federal + FICA table that is
approximate and not tax advice.

Career features you can use in any file:

- `probability`: chance an event happens at all.
- `group`: mutually exclusive outcomes, such as acquired vs shut down.
- `same_time_as`: linked events, which happen only if their anchor does.
- `duration`: a temporary change, such as school, that then reverts.
- `loan`: cash now, interest accrues, then repayment after a deferral.
- `spread`: a lognormal amount, such as exit proceeds.
- Per-stream `noise`: commission or bonus swings that do not compound.

## Run

```powershell
python -m stats_projections "scenarios/*.yaml" --chart outputs/compare.png --csv outputs
```

The first scenario (or the file named `baseline` within a glob) is the
reference the others are compared against. Options: `--paths N`, `--seed N`.

Output, for each scenario:

- **P(fail)**: share of paths where the portfolio drops below the floor at
  any point, even briefly, with a 95% confidence interval.
- **Ends below**: share still below the floor at the end. A big gap between
  this and P(fail) means a temporary cash crunch (debt you recover from),
  not a broke retirement.
- **Median fail**: typical age of the first failure.
- **P10 / P50 / P90 end**: the spread of final balances.
- **Versus baseline**: the change in failure rate with a paired confidence
  interval. Scenarios share their random draws, so this measures the
  decision, not noise.

## Sensitivity analysis

Which input drives the failure rate most, so you know what to estimate
carefully:

```powershell
python -m stats_projections scenarios/baseline.yaml --tornado outputs/tornado.png
```

Each input is nudged down and up on its own, and the scenario is rerun on
the same random draws:

- **Dollar amounts:** ±10% (`--money-step`). This covers the starting
  balance, every income and expense, and every event amount.
- **Return and inflation means:** ±1 point (`--rate-step`).
- **Stream growth:** ±0.5 point.
- **Return volatility:** ±3 points.
- **Job-gap chance:** ±2 points.
- **Event timing:** ±2 years (`--year-step`). Linked events move with
  their anchor.

The table and tornado chart rank inputs by **swing**, the biggest change in
P(fail) either way. `--sensitivity` prints the table without a chart, and
`--top N` limits how many inputs are shown. With several scenarios, each one
gets its own table and chart.

Compare swings only within the same nudge type. A 10% change in salary
and a 1-point change in returns are different-sized questions.

## Scenario files

The example numbers in `scenarios/` are placeholders. Replace them with yours.

- Money is in **today's dollars**. Income is **take-home** pay. Whatever is
  left after expenses is saved into the portfolio.
- **Year 0** is the first simulated year. A stream is active for
  `start_year <= year < end_year`.
- Uncertain inputs take `growth: {mean, stdev}`. Income can have
  `loss_probability` / `loss_fraction` for job gaps.
- Event types: `purchase`, `windfall`, `set_amount` (change a stream from a
  given year; 0 ends it). `jitter: N` shifts the year by up to N either way.
  `same_time_as` + `offset` ties events together.
- `extends: baseline.yaml` inherits everything. Lists merge by `name`; add
  `remove: true` to drop an inherited item. Unknown keys are errors, so
  typos fail loudly.

Return models (`portfolio.returns.model`):

- `normal`: mean and stdev (nominal).
- `student_t`: same mean and stdev with fatter tails (`dof`, default 5).
- `bootstrap`: resamples real history from a CSV you supply with columns
  `year,return,inflation` in decimals (0.07, not 7). Set
  `data: history.csv` and optionally `block_size: 5` to keep runs of years
  together. No data ships with the tool, so use a source you trust.

## Use from Python

```python
from stats_projections import load_scenario, simulate, summarize, compare
base = simulate(load_scenario("scenarios/baseline.yaml"))
home = simulate(load_scenario("scenarios/buy_home.yaml"))
print(summarize(base).p_fail, compare(base, home).delta)
```

## Layout

| Path | Role |
| --- | --- |
| `stats_projections/config.py` | Load, merge, and validate scenario YAML |
| `stats_projections/draws.py` | Keyed random draws (common random numbers) and return models |
| `stats_projections/engine.py` | Pure, vectorized simulation |
| `stats_projections/report.py` | Summaries, paired comparison, CSV, charts |
| `stats_projections/sensitivity.py` | One-at-a-time sensitivity and tornado chart |
| `stats_projections/edit.py` | Scenario to file data, and minimal override diffs |
| `app.py` | Streamlit front end |
| `profiles/`, `careers/`, `data/` | You, career paths, tax table |
| `stats_projections/cli.py` | Command line |
| `tests/` | `python -m pytest` |

## Not built yet (from the roadmap)

- Goal search.
- Separate 401(k), Roth, and taxable accounts. The portfolio is treated as
  after-tax, and Social Security is untaxed.
- Monthly steps.
- Location-specific cost of living for each career.
# Personal-Finance-Projections
