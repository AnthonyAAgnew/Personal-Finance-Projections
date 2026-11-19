"""The simulation engine: a scenario in, net worth paths out.

Pure functions only: no files written, no printing, no plotting. All paths are
simulated at once as NumPy arrays shaped (paths, years).

Each year t, with real portfolio return r_t and net cash flow c_t
(take-home income - expenses - purchases - loan payments + windfalls + loans):

    W_{t+1} = W_t * (1 + r_t) + c_t        if W_t >= 0
    W_{t+1} = W_t * (1 + borrowing_rate) + c_t   otherwise

Cash flows land at the end of the year, which is slightly conservative for
savers. The real return is (1 + nominal) / (1 + inflation) - 1. Taxable
(gross) income is taxed with the scenario's progressive tax layers, applied
to the total taxable income of each path and year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import draws
from .config import CONFIDENCE, Event, Scenario, Stream, TaxLayer

NEVER = 10**9  # event year on paths where the event does not happen


@dataclass
class Result:
    scenario: Scenario
    # (paths, years + 1): portfolio minus loans owed. Column 0 is the start.
    # Failure is judged on the portfolio alone (can you pay your bills?).
    net_worth: np.ndarray
    cash_flow: np.ndarray  # (paths, years)
    real_returns: np.ndarray  # (paths, years)
    failed: np.ndarray  # (paths,) bool
    first_failure_year: np.ndarray  # (paths,) int, -1 if never
    income: np.ndarray | None = None  # (paths, years) all income before tax
    tax: np.ndarray | None = None  # (paths, years)
    take_home: np.ndarray | None = None  # (paths, years) spendable income after tax
    spending: np.ndarray | None = None  # (paths, years) all budget spending
    tier: np.ndarray | None = None  # (paths, years) lifestyle tier index, 0 = base
    affordable: np.ndarray | None = None  # (paths, years) income the tiers are judged on
    flow_means: dict | None = None  # mean of every flow by year, for the Sankey
    portfolio: np.ndarray | None = None  # (paths, years + 1) investable assets (W)
    debt: np.ndarray | None = None  # (paths, years + 1) student and other loans owed
    hours: np.ndarray | None = None  # (paths, years) work hours per week
    strategy_year: np.ndarray | None = None  # (paths,) year FIRE / coast began, -1 never
    cash: np.ndarray | None = None  # (paths, years + 1) cash reserve (part of portfolio)
    home: np.ndarray | None = None  # (paths, years + 1) home value (in net worth only)

    @property
    def p_fail(self) -> float:
        return float(self.failed.mean())


# ----------------------------------------------------------------------------
# Events: when (and whether) each one happens on each path
# ----------------------------------------------------------------------------


def event_years(s: Scenario) -> dict[str, np.ndarray]:
    """Per path year of each event, after outcomes, jitter and links.

    Paths where an event does not happen get ``NEVER``. A linked event
    inherits that, so it only happens where its anchor did.
    """
    by_name = {e.name: e for e in s.events}
    # Members of an outcome group split one uniform draw into disjoint slices.
    group_offsets: dict[str, float] = {}
    slices: dict[str, tuple[float, float]] = {}
    for e in s.events:
        if e.group:
            start = group_offsets.get(e.group, 0.0)
            group_offsets[e.group] = start + e.probability
            slices[e.name] = (start, start + e.probability)

    years: dict[str, np.ndarray] = {}

    def happens(e: Event) -> np.ndarray | None:
        if e.group:
            lo, hi = slices[e.name]
            u = draws.rng_for(s.seed, "group", e.group).random(s.paths)
            return (u >= lo) & (u < hi)
        if e.probability < 1:
            return draws.rng_for(s.seed, "chance", e.name).random(s.paths) < e.probability
        return None

    def resolve(e: Event, stack: tuple[str, ...] = ()) -> np.ndarray:
        if e.name in years:
            return years[e.name]
        if e.name in stack:
            raise ValueError(f"circular same_time_as: {' -> '.join((*stack, e.name))}")
        if e.same_time_as:
            y = resolve(by_name[e.same_time_as], (*stack, e.name)) + e.offset
        else:
            y = np.full(s.paths, e.year + e.offset, dtype=np.int64)
        if e.jitter:
            rng = draws.rng_for(s.seed, "jitter", e.name)
            y = y + rng.integers(-e.jitter, e.jitter + 1, s.paths)
        y = np.maximum(y, 0)
        occurs = happens(e)
        if occurs is not None:
            y = np.where(occurs, y, NEVER)
        years[e.name] = np.minimum(y, NEVER)
        return years[e.name]

    for e in s.events:
        resolve(e)
    return years


def nominal_year(s: Scenario, e: Event) -> int:
    by_name = {x.name: x for x in s.events}
    total = e.offset
    while e.same_time_as:
        e = by_name[e.same_time_as]
        total += e.offset
    return e.year + total


def equity_proceeds(s: Scenario, e: Event) -> np.ndarray:
    """After-tax founder proceeds per path from the event's cap table."""
    eq = e.equity
    z = draws.rng_for(s.seed, "sale", e.name).standard_normal(s.paths)
    sigma = float(np.hypot(eq.sale_spread, CONFIDENCE[e.confidence]))
    price = eq.sale_median * np.exp(sigma * z)
    investors = 1.0 - eq.common
    converts = investors * price >= eq.raised
    as_common = eq.stake * price
    after_pref = (eq.stake / eq.common if eq.common > 0 else 0.0) * np.maximum(price - eq.raised, 0)
    gross = np.where(converts, as_common, after_pref)
    tax = eq.state_rate * gross + eq.federal_rate * np.maximum(gross - eq.qsbs_cap, 0.0)
    return gross - tax


def event_amounts(s: Scenario, e: Event) -> np.ndarray:
    """Per path amount: fixed, or lognormal with median ``amount``.

    ``spread`` and the event's confidence level both widen it; a windfall
    with ``equity`` is worked out from the cap table instead.
    """
    if e.equity is not None:
        return equity_proceeds(s, e)
    sigma = float(np.hypot(e.spread, CONFIDENCE[e.confidence]))
    if not sigma:
        return np.full(s.paths, e.amount)
    z = draws.rng_for(s.seed, "spread", e.name).standard_normal(s.paths)
    return e.amount * np.exp(sigma * z)


# ----------------------------------------------------------------------------
# Streams
# ----------------------------------------------------------------------------


def stream_amounts(
    s: Scenario, stream: Stream, ev_years: dict[str, np.ndarray]
) -> np.ndarray:
    """Yearly amount of one stream on every path, shaped (paths, years)."""
    return _stream(s, stream, ev_years)[0]


def _stream(
    s: Scenario, stream: Stream, ev_years: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Amounts plus a mask of years the stream is off: inactive, or set to 0
    by an event (school pausing the budget, retirement ending pay)."""
    P, T = s.paths, s.horizon_years
    t = np.arange(T)

    g = stream.growth.mean + stream.growth.stdev * draws.standard_normal(
        s.seed, P, T, "growth", stream.name
    )
    g[:, 0] = 0.0  # the stated amount applies in its first year
    # G[p, t]: cumulative growth from year 0 to year t.
    G = np.cumprod(1.0 + np.maximum(g, -0.99), axis=1)

    start = min(stream.start_year, T - 1)
    level = stream.amount * G / G[:, [start]]

    changes = [e for e in s.events if e.type == "set_amount" and e.target == stream.name]
    rows = np.arange(P)
    paused = np.zeros((P, T), dtype=bool)
    for e in sorted(changes, key=lambda e: nominal_year(s, e)):
        y = ev_years[e.name]
        on = y < T
        G_at = G[rows, np.minimum(y, T - 1)][:, None]
        new = event_amounts(s, e)[:, None] * G / G_at
        mask = (t[None, :] >= y[:, None]) & on[:, None]
        if e.duration is not None:
            mask &= t[None, :] < (y + e.duration)[:, None]
        level = np.where(mask, new, level)
        paused = np.where(mask, e.amount == 0, paused)

    end = T if stream.end_year is None else stream.end_year
    active = (t >= stream.start_year) & (t < end)
    level = level * active[None, :]
    off = paused | ~active[None, :]

    # Scale events multiply whatever the level is (e.g. a remote-work pay cut,
    # cheaper living abroad), on top of any set_amount changes.
    for e in s.events:
        if e.type == "scale" and e.target == stream.name:
            y = ev_years[e.name]
            mask = (t[None, :] >= y[:, None]) & (y < T)[:, None]
            if e.duration is not None:
                mask &= t[None, :] < (y + e.duration)[:, None]
            level = np.where(mask, level * e.factor, level)

    sigma = CONFIDENCE[stream.confidence]
    if sigma:
        z = draws.rng_for(s.seed, "confidence", stream.name).standard_normal((P, 1))
        level = level * np.exp(sigma * z)
    if stream.noise > 0:
        z = draws.standard_normal(s.seed, P, T, "noise", stream.name)
        # Mean-preserving lognormal swing around the trend; it does not compound.
        level = level * np.exp(stream.noise * z - stream.noise**2 / 2)
    if stream.loss_probability > 0:
        hit = draws.uniform(s.seed, P, T, "shock", stream.name) < stream.loss_probability
        level = np.where(hit, level * (1.0 - stream.loss_fraction), level)
    return level, off


# ----------------------------------------------------------------------------
# Taxes and loans
# ----------------------------------------------------------------------------


def progressive_tax(income: np.ndarray, layers: tuple[TaxLayer, ...]) -> np.ndarray:
    tax = np.zeros_like(income)
    for layer in layers:
        base = np.maximum(income - layer.deduction, 0.0)
        bounds = [a for a, _ in layer.brackets] + [np.inf]
        for (lo, rate), hi in zip(layer.brackets, bounds[1:]):
            if rate:
                tax += rate * np.clip(base - lo, 0.0, hi - lo)
    return tax


def loan_payment(principal: np.ndarray, real_rate: float, defer: int, term: int) -> np.ndarray:
    """Equal yearly real payment, first one ``defer`` years after disbursement.

    The present value of the payments at disbursement equals the principal:
    interest accrues through the deferral, then payments start right away
    (an annuity due on the grown balance).
    """
    if real_rate == 0:
        return principal / term
    r = real_rate
    balance = principal * (1 + r) ** defer
    return balance * r / (1 - (1 + r) ** -term) / (1 + r)


# ----------------------------------------------------------------------------
# Cash flows and the main loop
# ----------------------------------------------------------------------------


@dataclass
class Flows:
    """Every money flow known before the yearly loop, each (paths, years)."""

    income: dict[str, np.ndarray]  # per income stream, before tax
    taxable: dict[str, bool]
    spendable: dict[str, bool]
    expenses: dict[str, np.ndarray]  # per expense stream, as written in the files
    off: dict[str, np.ndarray]  # expense stream inactive or paused by an event
    hours: dict[str, np.ndarray]  # per time stream, hours per week
    purchases: np.ndarray
    windfalls: np.ndarray  # after their flat tax
    loans_in: np.ndarray
    loan_payments: np.ndarray
    debt: np.ndarray  # (paths, years + 1) loan balance owed at the end of each year
    home_costs: np.ndarray  # property tax, insurance, upkeep
    home: np.ndarray  # (paths, years + 1) home value at the end of each year

    @property
    def total_income(self) -> np.ndarray:
        return sum(self.income.values()) if self.income else np.zeros_like(self.purchases)


def flows(s: Scenario) -> Flows:
    P, T = s.paths, s.horizon_years
    ev_years = event_years(s)
    zeros = lambda: np.zeros((P, T))  # noqa: E731
    income = {st.name: stream_amounts(s, st, ev_years) for st in s.income}
    expenses, off = {}, {}
    for st in s.expenses:
        expenses[st.name], off[st.name] = _stream(s, st, ev_years)
    hours = {st.name: stream_amounts(s, st, ev_years) for st in s.time}

    purchases, windfalls, loans_in, loan_payments = zeros(), zeros(), zeros(), zeros()
    home_costs = zeros()
    debt = np.zeros((P, T + 1))
    home = np.zeros((P, T + 1))
    rows = np.arange(P)

    def add_loan(amount, rate, defer, term, y):
        # Loan rates are nominal; payments are fixed in real terms using
        # the scenario's mean inflation (a transparent approximation).
        r = (1 + rate) / (1 + s.inflation.mean) - 1
        pay = loan_payment(amount, r, defer, term)
        for k in range(term):
            ty = y + defer + k
            ok = ty < T
            np.add.at(loan_payments, (rows[ok], ty[ok]), pay[ok])
        # Balance owed at the end of each year after disbursement: grows
        # with interest, falls with each payment, reaches 0 after the last.
        for j in range(defer + term):
            ty = y + j
            ok = ty < T
            grown = amount * (1 + r) ** j
            if j >= defer:
                n = j - defer + 1
                paid = pay * (((1 + r) ** n - 1) / r if r else n)
            else:
                paid = 0.0
            bal = np.maximum(grown - paid, 0.0)
            np.add.at(debt, (rows[ok], ty[ok] + 1), bal[ok])

    for e in s.events:
        if e.type in ("set_amount", "scale"):
            continue
        y = ev_years[e.name]
        on = y < T
        amount = event_amounts(s, e)
        if e.type == "purchase":
            np.add.at(purchases, (rows[on], y[on]), amount[on])
        elif e.type == "windfall":
            np.add.at(windfalls, (rows[on], y[on]), amount[on] * (1 - e.tax_rate))
        elif e.type == "loan":
            np.add.at(loans_in, (rows[on], y[on]), amount[on])
            add_loan(amount, e.rate, e.defer, e.term, y)
        elif e.type == "home":
            # Down payment and ~2% closing costs in cash; the rest is a
            # mortgage paid to the seller (no cash in), repaid like a loan.
            np.add.at(purchases, (rows[on], y[on]), (amount * (e.down + 0.02))[on])
            if e.down < 1:
                add_loan(amount * (1 - e.down), e.rate, 0, e.term, y)
            g = e.appreciation.mean + e.appreciation.stdev * draws.standard_normal(
                s.seed, P, T, "home", e.name)
            value = np.zeros(P)
            for j in range(T):
                owned = on & (j >= y)
                value = np.where(j == y, amount, value * (1 + np.where(owned, g[:, j], 0.0)))
                value = np.where(owned, value, 0.0)
                home[:, j + 1] += value
                home_costs[:, j] += np.where(owned, amount * e.owner_costs, 0.0)
    return Flows(income, {st.name: st.taxable for st in s.income},
                 {st.name: st.spendable for st in s.income}, expenses, off, hours,
                 purchases, windfalls, loans_in, loan_payments, debt, home_costs, home)


def stock_share(s: Scenario) -> np.ndarray | None:
    """Stock share by year from the glide path, or None (all in returns)."""
    if not s.allocation:
        return None
    pts = sorted(s.allocation)
    return np.interp(np.arange(s.horizon_years), [y for y, _ in pts], [x for _, x in pts])


def market_returns(s: Scenario) -> tuple[np.ndarray, np.ndarray | None]:
    """Real portfolio returns and real cash returns, each (paths, years).

    Stress tests add to stock returns and inflation in their years. With a
    glide path, the portfolio blends stocks with bonds correlated to them.
    """
    P, T = s.paths, s.horizon_years
    nominal, infl = draws.market(s.returns, s.inflation, s.seed, P, T)
    stock, infl = nominal.copy(), infl.copy()
    for y, dr, di in s.stress:
        if 0 <= y < T:
            stock[:, y] += dr
            infl[:, y] += di
    port = stock
    share = stock_share(s)
    if share is not None and s.bonds is not None:
        sd = s.returns.stdev if s.returns.stdev > 0 else 1.0
        z = (nominal - s.returns.mean) / sd
        e = draws.standard_normal(s.seed, P, T, "market", "bonds")
        rho = s.bonds.correlation
        bond = s.bonds.mean + s.bonds.stdev * (rho * z + np.sqrt(1 - rho**2) * e)
        port = share * stock + (1 - share) * bond
    real = (1.0 + np.maximum(port, -0.99)) / (1.0 + infl) - 1.0
    cash = None
    if s.cash is not None:
        cn = s.cash.mean + s.cash.stdev * draws.standard_normal(s.seed, P, T, "market", "cash")
        cash = (1.0 + np.maximum(cn, -0.5)) / (1.0 + infl) - 1.0
    return real, cash


def real_returns(s: Scenario) -> np.ndarray:
    return market_returns(s)[0]


def reserve_months(s: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Months of spending to hold in cash each year, and the runway years."""
    T = s.horizon_years
    months = np.zeros(T)
    runway = np.zeros(T, dtype=bool)
    if s.cash is not None:
        months[:] = s.cash.reserve_months
        if s.cash.runway_window and s.cash.runway_months > 0:
            a, b = s.cash.runway_window
            runway[max(a, 0):max(min(b, T), 0)] = True
            months[runway] = s.cash.runway_months
    return months, runway


CREEP = "Lifestyle creep"
COAST = "Coast spending (not saved)"


def _layer_tax(amount: np.ndarray, layer: TaxLayer) -> np.ndarray:
    return progressive_tax(amount, (layer,))


def simulate(s: Scenario) -> Result:
    P, T = s.paths, s.horizon_years
    r, rc = market_returns(s)
    F = flows(s)
    months, runway = reserve_months(s)
    ls = s.lifestyle
    cats = ls.categories if ls and ls.tiers else ()
    st = s.strategy if s.strategy and s.strategy.kind != "work" else None
    stop = set(st.stop) if st else set()

    # Spending outside the lifestyle tiers, and one-off flows, are fixed.
    fixed_spend = np.zeros((P, T))
    for name, a in F.expenses.items():
        if name not in cats:
            fixed_spend += a
    events_net = F.windfalls + F.loans_in - F.purchases - F.loan_payments - F.home_costs
    hours_base = sum(F.hours.values()) if F.hours else np.zeros((P, T))
    hours_stop = sum((a for n, a in F.hours.items() if n in stop), np.zeros((P, T)))

    if cats:
        thresholds = np.array([t.take_home for t in ls.tiers])
        tier_amt = {c: np.array([np.nan] + [12 * dict(t.monthly).get(c, np.nan) for t in ls.tiers])
                    for c in cats}
        cost_index = ls.index_by_year(T)
        any_on = np.zeros((P, T), dtype=bool)
        for c in cats:
            any_on |= ~F.off[c]

    # Outputs filled year by year.
    W = np.empty((P, T + 1))  # invested
    W[:, 0] = s.starting_balance
    C = np.zeros((P, T + 1))  # cash reserve (never negative)
    if s.cash is not None:
        C[:, 0] = s.cash.balance
    coast_spend = np.zeros((P, T))
    coasting = np.zeros(P, dtype=bool)  # true Coast FIRE: stopped saving
    stop_saving = st is not None and st.kind == "coast" and st.coast_mode == "stop_saving"
    net = np.empty((P, T))
    take_home = np.empty((P, T))
    income_total = np.empty((P, T))
    tax_total = np.empty((P, T))
    hours = np.empty((P, T))
    spend = {c: np.empty((P, T)) for c in cats}
    creep = np.zeros((P, T))
    tier = np.zeros((P, T), dtype=np.int8)
    afford = np.zeros((P, T)) if cats else None
    inc_means = {n: np.zeros(T) for n in F.income}
    tax_means = {t.name: np.zeros(T) for t in s.taxes}
    history: list[np.ndarray] = []  # recent take-home, for smoothing

    factor = np.ones(P)  # 1 = working normally; FIRE -> 0; coast -> coast_factor
    fire_year = np.full(P, -1)
    peak_spend = np.zeros(P)  # strategies plan around the most you have spent so far

    for t in range(T):
        w = W[:, t] + C[:, t]  # liquid assets
        # Income this year, with the strategy applied to the "stop" streams.
        inc_t = np.zeros(P)
        taxable_t = np.zeros(P)
        spendable_t = np.zeros(P)
        per_stream_taxable = []
        for name, a in F.income.items():
            v = a[:, t] * factor if name in stop else a[:, t]
            inc_means[name][t] = v.mean()
            inc_t += v
            if F.taxable[name]:
                taxable_t += v
                per_stream_taxable.append(v)
            if F.spendable[name]:
                spendable_t += v
        tax_t = np.zeros(P)
        for layer in s.taxes:
            if layer.per_earner:
                lt = sum((_layer_tax(v, layer) for v in per_stream_taxable), np.zeros(P))
            else:
                lt = _layer_tax(taxable_t, layer)
            tax_means[layer.name][t] = lt.mean()
            tax_t += lt
        th_t = spendable_t - tax_t
        take_home[:, t], income_total[:, t], tax_total[:, t] = th_t, inc_t, tax_t
        hours[:, t] = hours_base[:, t] - (1.0 - factor) * hours_stop[:, t]

        c_t = inc_t - tax_t + events_net[:, t] - fixed_spend[:, t]
        spent = fixed_spend[:, t].copy()
        if cats:
            history.append(th_t)
            avg = np.mean(history[-ls.smoothing_years:], axis=0)
            draw = ls.withdrawal_rate * np.maximum(w, 0.0)
            if st is not None:
                # People chasing FIRE don't spend from the portfolio while
                # still working; the draw starts once they stop or coast.
                draw = np.where(factor < 1.0, draw, 0.0)
            a_t = avg + draw
            afford[:, t] = a_t
            k = np.searchsorted(thresholds, a_t, side="right")
            if ls.max_tier is not None:
                k = np.minimum(k, ls.max_tier)
            tier[:, t] = k
            for c in cats:
                amt = tier_amt[c][k] * cost_index[t]
                base = F.expenses[c][:, t]
                val = np.where(np.isnan(amt) | F.off[c][:, t], base, amt)
                spend[c][:, t] = val
                c_t -= val
                spent += val
            above = a_t - np.where(k > 0, thresholds[np.maximum(k - 1, 0)], np.inf)
            creep[:, t] = np.where((k > 0) & any_on[:, t], ls.creep * np.maximum(above, 0.0), 0.0)
            c_t -= creep[:, t]
            spent += creep[:, t]
        if stop_saving:
            extra = np.where(coasting, np.maximum(c_t, 0.0), 0.0)
            coast_spend[:, t] = extra
            c_t = c_t - extra
            spent = spent + extra
        net[:, t] = c_t
        inv = W[:, t]
        inv_g = inv * np.where(inv >= 0, 1.0 + r[:, t], 1.0 + s.borrowing_rate)
        if s.cash is None:
            W[:, t + 1] = inv_g + c_t
        else:
            cash = C[:, t] * (1.0 + rc[:, t])
            target = months[t] / 12.0 * spent
            gap = np.maximum(target - cash, 0.0)
            to_cash = np.where(c_t > 0, np.minimum(c_t, gap), 0.0)
            from_cash = np.where(c_t < 0, np.minimum(-c_t, cash), 0.0)
            cash = cash + to_cash - from_cash
            inv_g = inv_g + c_t - to_cash + from_cash
            if runway[t]:
                # Building or living on a runway: sell investments to fill it.
                top = np.minimum(np.maximum(target - cash, 0.0), np.maximum(inv_g, 0.0))
            else:
                # Otherwise excess cash above the reserve is invested.
                top = -np.maximum(cash - target, 0.0)
            W[:, t + 1] = inv_g - top
            C[:, t + 1] = cash + top

        # Strategy: decide for next year from the year-end portfolio.
        peak_spend = np.maximum(peak_spend, spent)
        yrs = st.target_year - (t + 1) if st is not None else 0
        if st is not None and t + 1 >= st.min_year and yrs > 0:
            target = st.multiple * peak_spend
            liquid = W[:, t + 1] + C[:, t + 1]
            if st.kind == "fire":
                hit = (factor > 0) & (liquid >= target)
                factor = np.where(hit, 0.0, factor)
            elif stop_saving:
                need = target / (1 + st.expected_real_return) ** yrs
                hit = ~coasting & (liquid >= need)
                coasting |= hit
            else:  # coast, reduced pay and hours
                need = target / (1 + st.expected_real_return) ** yrs
                hit = (factor == 1.0) & (liquid >= need)
                factor = np.where(hit, st.coast_factor, factor)
            fire_year = np.where(hit & (fire_year < 0), t + 1, fire_year)

    W = W + C  # the portfolio reported everywhere is all liquid assets
    below = W[:, 1:] < s.failure_floor
    failed = below.any(axis=1)
    first = np.where(failed, below.argmax(axis=1), -1)

    out_spend = {n: a for n, a in F.expenses.items() if n not in cats}
    out_spend.update(spend)
    if cats:
        out_spend[CREEP] = creep
    if stop_saving:
        out_spend[COAST] = coast_spend
    spending = sum(out_spend.values()) if out_spend else np.zeros((P, T))
    means = {
        "income": inc_means,
        "tax": tax_means,
        "spending": {k: v.mean(axis=0) for k, v in out_spend.items()},
        "other": {"One-time purchases": F.purchases.mean(axis=0),
                  "Loan payments": F.loan_payments.mean(axis=0),
                  "Windfalls (after tax)": F.windfalls.mean(axis=0),
                  "Loans received": F.loans_in.mean(axis=0),
                  "Home costs": F.home_costs.mean(axis=0)},
        "saved": net.mean(axis=0),
    }
    return Result(s, W + F.home - F.debt, net, r, failed, first, income_total, tax_total,
                  take_home=take_home, spending=spending,
                  tier=tier if cats else None, affordable=afford, flow_means=means,
                  portfolio=W, debt=F.debt, hours=hours if s.time else None,
                  strategy_year=fire_year if st else None,
                  cash=C if s.cash is not None else None,
                  home=F.home if F.home.any() else None)


def cash_flows(s: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Net cash flow, income before tax, and tax, as simulated."""
    res = simulate(s)
    return res.cash_flow, res.income, res.tax
