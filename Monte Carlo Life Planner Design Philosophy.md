# Monte Carlo Life Planner: Design Philosophy

Nov 18, 2026 · @Anthony Agnew

## Purpose

This tool answers one question: across thousands of plausible futures, how often do my finances fail? It replaces a heavyweight licensed planner with a small, personal simulator that is fast to change and easy to trust.

The user enters expected income, expenses, savings, and major planned purchases, then tries alternative timelines such as a career change, a home purchase, or an earlier retirement. The simulator runs a Monte Carlo analysis on each scenario and reports the probability of complete failure, meaning the portfolio hits zero before the end of the planning horizon, alongside the spread of likely outcomes.

## Design philosophy

1. **Small, pure core.** The simulation engine is a set of plain functions with no UI, file, or network code inside it. Same inputs and seed give the same outputs.
2. **Configuration over code.** Every personal assumption lives in a readable scenario file. Trying a new career or purchase means editing a file, never the engine.
3. **Uncertainty is explicit.** Every input that is not known for certain is a distribution with a stated rationale, not a single point guess.
4. **Scenarios are first class.** Comparing futures side by side is the main use, so scenarios share a baseline and override only what differs.
5. **Transparent over clever.** Prefer a simple model you can audit over a sophisticated one you cannot explain. Add realism only when it changes a decision.
6. **Vectorized from day one.** Simulate all paths at once as NumPy arrays so 10,000 paths over 50 years runs in seconds.

## Core requirements

Version 1 is done when it can do all of the following.

- **Cash flow model:** yearly (or monthly) income, recurring expenses, and savings contributions, each with growth and volatility.
- **Discrete events:** one off purchases (car, home down payment, wedding) and step changes (new job, raise, career switch, retirement) placed at a year, with optional timing uncertainty.
- **Market returns:** stochastic portfolio returns from a stated distribution, plus inflation, with all reporting in real (inflation adjusted) dollars.
- **Monte Carlo engine:** at least 10,000 paths per scenario, reproducible with a fixed random seed.
- **Failure metric:** probability that net worth drops below zero (or a user set floor) at any point, and the year it first happens.
- **Outcome bands:** 10th, 50th, and 90th percentile net worth over time.
- **Scenario comparison:** run two or more scenarios against the same random draws so differences reflect the decision, not noise.

## Nice to haves

| Feature | Why it matters | Effort |
| --- | --- | --- |
| Sensitivity analysis | Shows which input drives failure risk most, so you know what to estimate carefully | Medium |
| Correlated returns | Stocks, bonds, and inflation move together; independent draws understate bad years | Medium |
| Historical bootstrap | Resamples real return sequences instead of assuming a normal distribution | Low |
| Tax aware accounts | Separates taxable, 457(b), and Roth balances with simple withdrawal rules | High |
| Income shocks | Random job loss or gap years with a set annual probability | Low |
| Web front end | Sliders and charts in Streamlit for quick what if exploration | Medium |
| Goal search | Solves for the savings rate or retirement age that hits a target success rate | Medium |

## Architecture

Three layers, each testable on its own: the config describes a future, the engine turns it into net worth paths, and reporting turns paths into decisions.

&#91;embedded content: architecture · 3 layers plus shared random draws\]

Because the engine holds no UI code, a terminal script, a Jupyter notebook, and a later Streamlit app all call the same functions. The shared random draws are what make scenario comparisons fair.

## Modeling notes and risks

The math is easy; the inputs are where the model can mislead you.

- **Fat tails:** normal returns understate crashes. Default to bootstrapping historical annual returns, with a normal or Student t option for comparison.
- **Sequence risk:** a crash right before a big purchase matters far more than one later. Year by year simulation captures this automatically, so never shortcut to average returns.
- **Common random numbers:** draw the return matrix once and reuse it across scenarios. This makes scenario differences far more precise for the same path count.
- **Convergence:** report a confidence interval on the failure probability. With 10,000 paths, a 5% failure rate carries roughly plus or minus 0.4 points.
- **False precision:** a 3.2% failure rate is a statement about your assumptions, not the future. Pair every headline number with the sensitivity of that number to its key inputs.

## Reference repositories

Study these for patterns rather than forking one wholesale; none models career and purchase events the way this tool needs.

| Repository | What to borrow |
| --- | --- |
| [IDwiiX/retirement-simulator](https://github.com/IDwiiX/retirement-simulator) | Output design: percentile bands, probability of not running out, Streamlit front end |
| [cakesandcode/monte-carlo-retirement](https://github.com/cakesandcode/monte-carlo-retirement) | Tax aware withdrawals, inflation, and Social Security modeling in Streamlit |
| [mattk1205/monte\_carlo\_retirement](https://github.com/mattk1205/monte_carlo_retirement) | YAML scenario configs, a CLI, profiling, and sensitivity runs |
| [pinireisman/retirement\_simulation](https://github.com/pinireisman/retirement_simulation) | Real dollar modeling with properties and one off lump sums |
| [jhykes/retirement-mc](https://github.com/jhykes/retirement-mc) | A small, focused core loop that is easy to read |

## Build roadmap

1. **Core loop (weekend 1):** one scenario file, normal returns, fixed income and expenses, failure probability printed to the terminal.
2. **Events and comparison (weekend 2):** purchases and career changes, common random numbers, percentile band chart.
3. **Realism pass:** historical bootstrap, inflation, income shocks, confidence interval on failure rate.
4. **Insight pass:** sensitivity analysis and goal search.
5. **Front end:** Streamlit app over the same engine, only once the engine is trusted.

Sources: [IDwiiX/retirement-simulator](https://github.com/IDwiiX/retirement-simulator), [cakesandcode/monte-carlo-retirement](https://github.com/cakesandcode/monte-carlo-retirement), [mattk1205/monte\_carlo\_retirement](https://github.com/mattk1205/monte_carlo_retirement), [pinireisman/retirement\_simulation](https://github.com/pinireisman/retirement_simulation), [jhykes/retirement-mc](https://github.com/jhykes/retirement-mc)
