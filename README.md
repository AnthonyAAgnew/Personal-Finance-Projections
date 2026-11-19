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

### Budget, lifestyle tiers, and milestones

- **Category budget:** the profile lists expenses by category, in monthly
  today's dollars (`monthly: 1400`).
- **Lifestyle tiers:** `lifestyle.tiers` set a monthly budget per category
  once your *affordable income* reaches a tier. Affordable income is your
  3-year average take-home plus `withdrawal_rate` x your portfolio.
- **Lifestyle creep:** between tiers you also spend `creep` of every extra
  dollar.
- **Paused categories:** an event that sets a category to 0 (for example,
  med school covering living costs) keeps it at 0 even inside a tier.
- **Milestones tab:**
  - take-home milestones ($200k / $250k / $300k), with the gross pay each
    needs;
  - net worth milestones ($1M to $30M), with the perpetual dividend each
    supports at the withdrawal rate.

  Both show the chance of ever reaching the milestone and the median age.
- **Cash flow tab:** a Sankey of one year's money, from income through
  taxes and budget categories into savings.
- **Employer matches** use `spendable: false`, so they go straight to
  savings and are not take-home.

### Money, markets and risk

- **Cash and investments:** the profile holds `portfolio.cash` (balance,
  reserve months) apart from investments. Savings refill the reserve first;
  shortfalls come out of cash first.
- **Glide path:** `portfolio.bonds` and `portfolio.allocation` blend stocks and
  bonds by year (90% stocks to 45, 60% at 65, 50% at 75 in the profile).
- **Market scenarios** (sidebar): lost decade, high-inflation decade, crash in
  your first career year, crash right after retiring, depression-level shock.
  Each lays fixed shocks over the same random paths (`stats_projections/stress.py`).
- **Three failure types:** cash crunch (the old P(fail)), negative net worth
  after 40, and running out after 65.
- **Startup runway:** founder careers set `cash.runway_window`; the life
  events "Startup runway: 4 / 2 years in cash" hold that much spending in cash
  through it. The Results tab shows whether cash or all liquid assets ran out.
- **Cap-table exits:** a windfall with `equity:` works out founder proceeds
  from the sale price, founder share, option pool, round dilution and 1x
  preferences, with QSBS-aware tax.
- **Confidence:** `confidence: medium | low` on a stream or event widens it
  (a persistent per-path level for streams). Assumptions marked CHECK show as
  low confidence in the app.
- **Home ownership:** `life/buy_home.yaml` uses the `home` event type: down
  payment, mortgage, owner costs and appreciation; the home counts in net worth.
- **Coast FIRE (stop saving)** is the standard definition; "Coast, half pay and
  hours" is the semi-retirement variant.
- **Tax accounts** (`accounts:` in the profile): while you have a paycheck,
  savings fill a 401(k) (pre-tax, with the employer match), an HSA and a Roth
  IRA at IRS 2025 limits, then a taxable account. Withdrawals come from taxable
  (capital gains tax), then the 401(k) (income tax, 10% penalty before 60),
  then Roth and HSA. The Results tab shows each account and an after-tax net
  worth at the chosen age. Not modeled: required minimum distributions, Roth
  conversions, Social Security taxation.
- **Decision points** (Decide tab): start on path A and, at a chosen age,
  switch to path B on the simulated lives that meet a rule (liquid savings or
  net worth above a threshold). Shows the share that switch and how the mixed
  plan compares with always A or always B (`stats_projections/decisions.py`).
- **Break-even finder** (Sensitivity tab): solves for the value of one input of
  option A (EP pay, exit price, tuition) that makes it match option B.
- **Free time** (Hours tab): waking hours left after work and commuting.

### Decide: weighted scores and Pareto view

- Eight criteria per option: net worth at 65 (median and P10), financial
  independence age (portfolio reaches 25x spending), years working for under
  $100k (school and training), lifetime work hours, autonomy, satisfaction and
  burnout.
- Each career file has a `quality:` line (autonomy from O*NET "Freedom to Make
  Decisions", satisfaction and burnout as shares from surveys), with sources.
- Weight sliders give a 0-100 score (relative to the options shown). The Pareto
  chart plots a financial score against a life score, or any two criteria,
  and marks the options nothing else beats on both.

### Life events, strategies, hours and debt

- **Three ways to compare:**
  - **Careers:** every career under one strategy.
  - **Strategies:** one career under every strategy.
  - **Life scenarios.**
- **Life events (`life/`):**
  - `partner.yaml`: a dual income with joint taxes (`data/taxes_ca_mfj.yaml`).
  - `kids_1.yaml` and `kids_2.yaml`: daycare, everyday costs, and college.
  - `expat_southern_europe.yaml`: remote US work from rural Italy or Spain.

  Pick any combination; they are layered on every option.
- **Location (`locations/`):** Bay Area, San Diego, Boston, Minneapolis, or
  Texas instead of Irvine, from the start of your career. Each scales pay,
  rent, other costs and lifestyle tiers, and swaps California taxes for the
  new state's (tax layers merge by name, so `remove: true` drops one). In the
  app the location is applied after life events, so a partner's joint return
  keeps the new state's taxes; on the command line, list it last.
- **Net worth at checkpoints:** the Results tab shows median net worth (P25
  to P75) at ages 30, 35, 40, 45, 50, 60 and 65 for every option.
- **Finance strategies (`strategies/`):**
  - Work to 65.
  - Traditional FIRE: stop at 25x peak spending.
  - Lean FIRE: capped lifestyle.
  - Fat FIRE: 33x.
  - Coast FIRE: half pay and half hours once you can coast to 25x by 65.
  - Barista FIRE: 25% pay and hours.
- **Hours:** each career sets hours per week for each phase from survey data (sources in each file and in the Hours tab), with one age taper (-5% at 50 and at 58) for everyone. Events
  and strategies change. The Hours tab shows hours by age, lifetime hours,
  average hours per week by decade, and take-home per work hour.
- **Debt:** loans are tracked as balances. Net worth = portfolio minus loans
  owed. The scorecard shows peak loans and the age they're paid off.
  `doctor.yaml` and `doctor_no_debt.yaml` compare medical school with and
  without loans.
- **From the command line:**

  ```powershell
  python -m stats_projections --profile profiles/my_profile.yaml "careers/*.yaml" --add life/partner.yaml --add strategies/coast_fire.yaml
  ```

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
| `doctor_no_debt.yaml` | Physician with a full scholarship | Same as `doctor.yaml`, no loans |
| `physician_cmo.yaml` | Physician, then industry medical director or CMO | Getinge and device CMO postings |
| `physician_academic.yaml` | Academic medicine | Medscape minus an academic discount (CHECK) |
| `patent_prosecution.yaml` | JD, then patent prosecution at an IP firm | ABA debt, AIPLA medians via LeanLaw |
| `patent_litigation.yaml` | JD, then BigLaw patent litigation | 2025 Cravath scale |
| `patent_inhouse.yaml` | JD, then in-house patent counsel (med device or pharma) | AIPLA via LeanLaw (CHECK) |
| `postdoc_academia.yaml` | Postdoc, then a faculty job or industry | NIH NRSA FY2025 stipends |
| `product_design.yaml` | Product design engineer | BLS mechanical engineers |
| `data_analytics.yaml` | Data scientist or analytics | BLS OEWS (CHECK) |
| `tech_ai_health.yaml` | Tech or AI health company | Typical big-tech levels (CHECK) |
| `startup_employee.yaml` | Early-stage startup employee | Carta option pools |
| `consumer_bootstrap.yaml` | Self-funded consumer product business | SBA and BLS survival rates |
| `quant_finance.yaml` | Quant researcher (NYC) | Jane Street and Citadel postings, TraderMath |
| `cardiology_ep.yaml` | MD, IM residency, cardiology and EP fellowships | MedAxiom 2026, Marit Health, Medscape 2025, NRMP 2025 |
| `industry_then_founder.yaml` | R&D engineer for ~4 years, then a self-funded device startup | Kruze, Citeline, PMCF |

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
