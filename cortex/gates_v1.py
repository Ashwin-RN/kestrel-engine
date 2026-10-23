"""Cortex hard boolean gates: value-trap detection in code, not in prompts.

Each gate is a pure function returning a GateResult. The stack's verdicts set
a tier CEILING: the maximum tier Cortex may assign a name. The language model
does the within-bound value judgement and is never allowed past the ceiling,
and the ceiling is a maximum, never an assignment. That is what makes the
boundary reproducible: the decision that can veto a name lives in
deterministic Python, not in model judgement.

Every gate reports one of three statuses (``gate_status``):

  PASS     the gate is affirmative and imposes nothing.
  FAIL     the gate measured the name and the reading is bad.
  ABSTAIN  the gate could not be evaluated (a source gap), so it does not
           vote. An ABSTAIN is never counted as a failure, except on a gate
           whose input is a mandatory disclosure, where absence is itself the
           red flag.

The ceiling is computed over gate FAMILIES, not raw gate counts, so two
correlated readings inside one family do not double-count against a name.
Tier B is reachable: a small number of soft misses, or a core pillar that could
not be measured, lands a name on B where Cortex can still weigh it, rather than
on the bottom of the ladder before the synthesizer ever sees it. The counts
that separate the tiers are withheld from the public edition.

Public edition: this is an excerpt. Every gate threshold is a module constant
set to None. A gate whose threshold is withheld ABSTAINS through
``_abstain_withheld`` (a distinct ``withheld:`` reason that every gate,
disclosure-required ones included, reads as ABSTAIN), so the stack still runs
end to end and the family policy is visible, but no value ships. The record
stamps, the measured-only valuation invariants and the rejected-pile floor are
not published.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

log = logging.getLogger("cortex.gates")


# --- Thresholds (withheld) ---------------------------------------------------
# The names say WHAT each gate measures; the calibrated values do not ship.

EARNINGS_QUALITY_MIN_RATIO: Optional[float] = None  # withheld from the public edition
DEBT_TRAJECTORY_MAX_ABSOLUTE: Optional[float] = None  # withheld from the public edition
DEBT_TRAJECTORY_1Y_TOLERANCE: Optional[float] = None  # withheld from the public edition
DEBT_TRAJECTORY_NEAR_ZERO_DE: Optional[float] = None  # withheld from the public edition
ROIIC_PROXY_MIN: Optional[float] = None  # withheld from the public edition
VALUATION_MIN_POSITIVE_PEERS: Optional[int] = None  # withheld from the public edition
VALUATION_MAX_PEER_PERCENTILE: Optional[float] = None  # withheld from the public edition
VALUATION_ABSOLUTE_PE_CEILING: Optional[float] = None  # withheld from the public edition
VALUATION_ROCE_ESCAPE: Optional[float] = None  # withheld from the public edition
VALUATION_EV_EBITDA_MAX_PEER_PERCENTILE: Optional[float] = None  # withheld from the public edition
VALUATION_EV_EBITDA_ABSOLUTE_CEILING: Optional[float] = None  # withheld from the public edition
VALUATION_EV_EBITDA_ROCE_ESCAPE: Optional[float] = None  # withheld from the public edition
# Distress age is severity-aware: the minimum months since the all-time high
# depend on the diagnosis severity, with a conservative default when severity
# is missing, and a maximum past which a non-recovery reads as structural.
DISTRESS_AGE_FLOOR_BY_SEVERITY: Optional[dict] = None  # withheld from the public edition
DISTRESS_AGE_DEFAULT_FLOOR: Optional[float] = None  # withheld from the public edition
DISTRESS_AGE_MAX_MONTHS: Optional[float] = None  # withheld from the public edition
PROMOTER_PLEDGE_MAX_PCT: Optional[float] = None  # withheld from the public edition
RECEIVABLES_VELOCITY_MULTIPLIER: Optional[float] = None  # withheld from the public edition
INTEREST_COVERAGE_MIN: Optional[float] = None  # withheld from the public edition
PEAK_MARGIN_MAX_MULTIPLIER: Optional[float] = None  # withheld from the public edition


# --- Result types ------------------------------------------------------------

@dataclass
class GateResult:
    """One gate's outcome.

    `passed=True` means the gate is affirmative (no ceiling imposed by it).
    `passed=False` is a FAIL or an ABSTAIN; ``gate_status`` tells them apart
    from the `why` prefix. `numeric_value` is the metric the gate looked at
    (None when data was missing). `why` is one line for the synth output and
    the dossier. `details` is an optional structured record of HOW a
    population-based verdict was reached (the two valuation gates name their
    peer cohort here); it never feeds `passed`, and readers must tolerate None.
    """
    name: str
    passed: bool
    why: str
    numeric_value: Optional[float] = None
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GatesEvaluation:
    """Full per-stock evaluation, computed once per dossier at the sweep and
    embedded in the view Cortex sees."""
    symbol: str
    results: list[GateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_gates(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict:
        # `all_passed` / `n_failed` / `failed_gates` keep their raw meaning
        # (every not-passed gate) so the validation ledger's gate vector stays
        # comparable across policy versions. The family / abstain view rides
        # as ADDITIONAL keys; the per-gate `status` is what the synthesizer
        # reads to weigh pillars nobody could measure.
        profile = gate_failure_profile(self)
        return {
            "symbol": self.symbol,
            "all_passed": self.all_passed,
            "n_failed": self.n_failed,
            "failed_gates": self.failed_gates,
            "gate_policy_version": GATE_POLICY_VERSION,
            "failed_families": sorted(profile.failed_families),
            "abstained_families": sorted(profile.abstained_families),
            "abstained": [
                {"name": r.name, "family": GATE_FAMILY.get(r.name), "why": r.why}
                for r in self.results if profile.statuses.get(r.name) == "ABSTAIN"
            ],
            "results": [
                {**r.to_dict(), "status": profile.statuses.get(r.name, gate_status(r))}
                for r in self.results
            ],
        }


# --- Shared result builders --------------------------------------------------

INSUFFICIENT_PREFIX = "insufficient_data:"
WITHHELD_PREFIX = "withheld:"


def _fail_insufficient_data(name: str, missing: str,
                            details: Optional[dict] = None) -> GateResult:
    """A source-limit non-answer. Raw `passed` stays False so legacy vectors
    keep their not-passed signal; ``gate_status`` reads it as ABSTAIN except on
    a disclosure-required gate, where missing data is a real FAIL."""
    return GateResult(name=name, passed=False, why=f"{INSUFFICIENT_PREFIX} {missing}",
                      numeric_value=None, details=details)


def _abstain_withheld(name: str, what: str) -> GateResult:
    """The public-edition guard: the gate's threshold is withheld, so it does
    not vote. Distinct from a data gap so a disclosure gate cannot read it as
    a FAIL."""
    log.debug("%s skipped: %s withheld from the public edition", name, what)
    return GateResult(name=name, passed=False,
                      why=f"{WITHHELD_PREFIX} {what} withheld from the public edition")


def _financial_na(name: str, metric: str) -> GateResult:
    """Banks, NBFCs, insurers and AMCs auto-pass the industrial-company tests:
    their statements are structurally different (NIM and NPA are their
    metrics)."""
    return GateResult(name=name, passed=True,
                      why=f"financial sector: {metric} not applicable (NIM/NPA are the metrics)")


def _cohort_details(peer_values: Optional[list], clean: list,
                    peer_selection: Optional[dict],
                    peer_symbols: Optional[list[str]]) -> dict:
    """Which cohort a valuation verdict was cast over. The cohort's IDENTITY
    is carried to the gate boundary so a reader can always name the peers,
    including on an ABSTAIN. Provenance only: nothing here moves `passed`."""
    sel = peer_selection if isinstance(peer_selection, dict) else {}
    used = sel.get("used")
    if isinstance(used, list) and used:
        symbols, source = [str(s) for s in used], "peer_selection.used"
    elif peer_symbols:
        symbols, source = [str(s) for s in peer_symbols if s], "peer_metrics_list"
    else:
        symbols, source = None, None
    return {"n_positive": len(clean), "n_cohort": len(peer_values or []),
            "cohort_method": sel.get("method"),
            "cohort_span_decades": sel.get("cap_span_decades"),
            "selector_version": sel.get("selector_version"),
            "cohort_symbols": symbols, "cohort_symbols_source": source}


# --- The gates ---------------------------------------------------------------

def passes_earnings_quality(*, ocf_5y_series: Optional[list[float]],
                            net_profit_5y_series: Optional[list[float]],
                            is_financial: bool = False) -> GateResult:
    """Accruals quality: multi-year average operating cash flow over average
    net profit. Earnings that never turn into cash are the classic trap."""
    name = "passes_earnings_quality"
    if is_financial:
        return _financial_na(name, "OCF/NP accruals test")
    if EARNINGS_QUALITY_MIN_RATIO is None:
        return _abstain_withheld(name, "accruals floor")
    if not ocf_5y_series or not net_profit_5y_series:
        return _fail_insufficient_data(name, "missing OCF or net profit series")
    if len(ocf_5y_series) < 3 or len(net_profit_5y_series) < 3:
        return _fail_insufficient_data(name, "fewer than 3 years of financials")
    np_avg = statistics.mean(net_profit_5y_series)
    if np_avg <= 0:
        # Have data, it is bad: sustained losses stay a FAIL, never an ABSTAIN.
        return GateResult(name, False, f"average net profit non-positive ({np_avg:.1f})")
    ratio = statistics.mean(ocf_5y_series) / np_avg
    passed = ratio >= EARNINGS_QUALITY_MIN_RATIO
    return GateResult(name, passed, f"OCF/NP = {ratio:.2f}", round(ratio, 3))


def passes_debt_trajectory(*, current_de: Optional[float], de_1y_ago: Optional[float],
                           de_3y_avg: Optional[float],
                           is_financial: bool = False) -> GateResult:
    """Debt-to-equity level AND direction: an absolute ceiling, then EITHER a
    one-year slope leg (catches a recent spike) OR a below-the-trailing-average
    leg (credits a name that is deleveraging on a messy path)."""
    name = "passes_debt_trajectory"
    if is_financial:
        return _financial_na(name, "D/E gate")
    if (DEBT_TRAJECTORY_MAX_ABSOLUTE is None or DEBT_TRAJECTORY_1Y_TOLERANCE is None
            or DEBT_TRAJECTORY_NEAR_ZERO_DE is None):
        return _abstain_withheld(name, "D/E ceiling and slope tolerance")
    if current_de is None:
        return _fail_insufficient_data(name, "missing current D/E")
    if de_1y_ago is None and de_3y_avg is None:
        return _fail_insufficient_data(name, "missing both 1Y-ago and 3Y avg D/E")
    if current_de > DEBT_TRAJECTORY_MAX_ABSOLUTE:
        return GateResult(name, False, f"absolute D/E={current_de:.2f} over the ceiling",
                          round(current_de, 3))
    slope_ok = (de_1y_ago is not None and de_1y_ago > 0
                and current_de < DEBT_TRAJECTORY_1Y_TOLERANCE * de_1y_ago)
    # A tiny base makes the slope meaningless; the absolute ceiling handles size.
    if de_1y_ago is not None and de_1y_ago <= DEBT_TRAJECTORY_NEAR_ZERO_DE:
        slope_ok = True
    avg_ok = de_3y_avg is not None and current_de < de_3y_avg
    return GateResult(name, slope_ok or avg_ok,
                      f"D/E={current_de:.2f}; slope_ok={slope_ok} avg_ok={avg_ok}",
                      round(current_de, 3))


def passes_roiic_proxy(*, net_profit_5y_series: Optional[list[float]],
                       net_worth_current: Optional[float],
                       net_worth_5y_ago: Optional[float],
                       is_financial: bool = False) -> GateResult:
    """Return on incremental invested capital, proxied as incremental net profit
    over incremental net worth: did retained capital earn anything? A floor for
    capital incinerators, set for a mean-reversion horizon rather than a
    compounder's cost of capital. Replaced an older net-worth-growth test,
    which a share issue parked in cash could pass."""
    name = "passes_roiic_proxy"
    if is_financial:
        return _financial_na(name, "ROIIC proxy")
    if ROIIC_PROXY_MIN is None:
        return _abstain_withheld(name, "incremental-return floor")
    if not net_profit_5y_series or len(net_profit_5y_series) < 5:
        return _fail_insufficient_data(name, "net_profit series < 5 years")
    if net_worth_current is None or net_worth_5y_ago is None:
        return _fail_insufficient_data(name, "net worth missing")
    inc_np = net_profit_5y_series[-1] - net_profit_5y_series[0]
    inc_nw = net_worth_current - net_worth_5y_ago
    if inc_nw <= 0:
        # Capital returned: judge by the direction of profit alone.
        return GateResult(name, inc_np > 0,
                          f"net worth shrank ({inc_nw:.1f}); incremental NP {inc_np:+.1f}")
    roiic = inc_np / inc_nw
    return GateResult(name, roiic >= ROIIC_PROXY_MIN, f"ROIIC proxy = {roiic:.1%}",
                      round(roiic, 4))


def _peer_relative_valuation(name: str, label: str, subject: Optional[float],
                             peers: Optional[list[float]], roce: Optional[float],
                             max_percentile: Optional[float],
                             absolute_ceiling: Optional[float],
                             roce_escape: Optional[float],
                             peer_selection: Optional[dict],
                             peer_symbols: Optional[list[str]]) -> GateResult:
    """The shared shape of both valuation gates: the subject's percentile in
    its peer cohort, AND an absolute ceiling with a return-on-capital escape
    for genuine compounders that legitimately trade at a premium. A thin
    cohort abstains rather than decides; the cohort is named on every path."""
    clean = [p for p in (peers or []) if p is not None and p > 0]
    details = _cohort_details(peers, clean, peer_selection, peer_symbols)
    if (max_percentile is None or absolute_ceiling is None or roce_escape is None
            or VALUATION_MIN_POSITIVE_PEERS is None):
        return _abstain_withheld(name, "percentile, ceiling and escape")
    if subject is None:
        return _fail_insufficient_data(name, f"stock {label} missing", details)
    if subject <= 0:
        return GateResult(name, False, f"stock {label} non-positive: cheapness unverifiable",
                          None, details)
    if len(clean) < VALUATION_MIN_POSITIVE_PEERS:
        return _fail_insufficient_data(name, f"only {len(clean)} positive {label} peers",
                                       details)
    # Percentile = share of peers strictly cheaper than the subject.
    percentile = sum(1 for p in clean if p < subject) / len(clean)
    relative_ok = percentile <= max_percentile
    abs_ok = subject < absolute_ceiling
    escape = roce is not None and roce > roce_escape
    passed = relative_ok and (abs_ok or escape)
    details.update(percentile=round(percentile, 3), relative_ok=relative_ok,
                   absolute_ok=abs_ok, escape_used=bool(escape and not abs_ok))
    return GateResult(
        name, passed,
        f"{label}={subject:.1f} at {percentile:.0%} of {len(clean)} positive peers "
        f"(rel_ok={relative_ok}); abs_ok={abs_ok} escape={escape}; "
        f"cohort={details.get('cohort_method') or 'unstamped'}",
        round(percentile, 3), details)


def passes_valuation_sector_relative(*, current_pe: Optional[float],
                                     peer_pe_list: Optional[list[float]],
                                     current_roce: Optional[float] = None,
                                     peer_selection: Optional[dict] = None,
                                     peer_symbols: Optional[list[str]] = None) -> GateResult:
    """P/E against the market-cap peer cohort (ROCE stands in for ROIC)."""
    return _peer_relative_valuation(
        "passes_valuation_sector_relative", "P/E", current_pe, peer_pe_list,
        current_roce, VALUATION_MAX_PEER_PERCENTILE, VALUATION_ABSOLUTE_PE_CEILING,
        VALUATION_ROCE_ESCAPE, peer_selection, peer_symbols)


def passes_valuation_ev_ebitda(*, current_ev_ebitda: Optional[float],
                               peer_ev_ebitda_list: Optional[list[float]],
                               current_roce: Optional[float] = None,
                               peer_selection: Optional[dict] = None,
                               peer_symbols: Optional[list[str]] = None) -> GateResult:
    """EV/EBITDA against the same cohort: capital-structure neutral, so a
    leveraged firm that looks cheap on P/E is caught here. Runs alongside the
    P/E gate, not instead of it."""
    return _peer_relative_valuation(
        "passes_valuation_ev_ebitda", "EV/EBITDA", current_ev_ebitda, peer_ev_ebitda_list,
        current_roce, VALUATION_EV_EBITDA_MAX_PEER_PERCENTILE,
        VALUATION_EV_EBITDA_ABSOLUTE_CEILING, VALUATION_EV_EBITDA_ROCE_ESCAPE,
        peer_selection, peer_symbols)


def passes_distress_age(*, months_from_ath: Optional[float],
                        severity_score: Optional[int] = None,
                        severity_votes: Optional[list] = None) -> GateResult:
    """Months since the all-time high must sit inside an accumulation window:
    too early is a falling knife, too late is structural decay. The floor is
    severity-aware. When several severity votes parse, the floor is lifted to
    the harshest vote's floor, so a straddling vote is never averaged past the
    cliff (it only ever raises the floor)."""
    name = "passes_distress_age"
    if (DISTRESS_AGE_FLOOR_BY_SEVERITY is None or DISTRESS_AGE_DEFAULT_FLOOR is None
            or DISTRESS_AGE_MAX_MONTHS is None):
        return _abstain_withheld(name, "accumulation window")
    if months_from_ath is None:
        return _fail_insufficient_data(name, "ATH date unknown")
    floor = (DISTRESS_AGE_DEFAULT_FLOOR if severity_score is None
             else DISTRESS_AGE_FLOOR_BY_SEVERITY.get(int(severity_score),
                                                     DISTRESS_AGE_DEFAULT_FLOOR))
    votes: list[int] = []
    for v in severity_votes if isinstance(severity_votes, list) else []:
        try:
            votes.append(int(v))
        except (TypeError, ValueError):
            continue
    if len(votes) >= 2:
        floor = max([floor] + [DISTRESS_AGE_FLOOR_BY_SEVERITY.get(v, DISTRESS_AGE_DEFAULT_FLOOR)
                               for v in votes])
    m = round(months_from_ath, 1)
    if months_from_ath < floor:
        return GateResult(name, False, f"{m}mo from ATH: too early for this severity", m)
    if months_from_ath > DISTRESS_AGE_MAX_MONTHS:
        return GateResult(name, False, f"{m}mo from ATH: structural decay", m)
    return GateResult(name, True, f"{m}mo from ATH: inside the accumulation window", m)


def passes_promoter_pledge(*, promoter_pledge_pct: Optional[float]) -> GateResult:
    """Promoter pledge ceiling. A heavily pledged promoter becomes a forced
    seller in a drawdown. Disclosure is mandatory for a listed Indian company,
    so a missing value is a FAIL (see DISCLOSURE_REQUIRED_GATES), not an
    ABSTAIN. A no-promoter company is recorded as zero and passes."""
    name = "passes_promoter_pledge"
    if PROMOTER_PLEDGE_MAX_PCT is None:
        return _abstain_withheld(name, "pledge ceiling")
    if promoter_pledge_pct is None:
        return _fail_insufficient_data(name, "pledge data missing (disclosure required)")
    passed = promoter_pledge_pct < PROMOTER_PLEDGE_MAX_PCT
    return GateResult(name, passed, f"promoter pledge={promoter_pledge_pct:.1f}%",
                      round(float(promoter_pledge_pct), 2))


def passes_receivables_velocity(*, revenue_3y_series: Optional[list[float]],
                                receivables_3y_series: Optional[list[float]]) -> GateResult:
    """Receivables growing much faster than revenue is the leading indicator of
    accounting-driven traps (channel stuffing, aggressive recognition). A
    receivables row the source does not break out is an ABSTAIN, never a
    PASS: a source limit is not proof of clean books."""
    name = "passes_receivables_velocity"
    if RECEIVABLES_VELOCITY_MULTIPLIER is None:
        return _abstain_withheld(name, "velocity multiple")
    if not revenue_3y_series or len(revenue_3y_series) < 3:
        return _fail_insufficient_data(name, "revenue series < 3 years")
    if not receivables_3y_series or len(receivables_3y_series) < 3:
        return _fail_insufficient_data(name, "receivables not broken out by the source")
    rev0, rev1 = revenue_3y_series[-3], revenue_3y_series[-1]
    rec0, rec1 = receivables_3y_series[-3], receivables_3y_series[-1]
    if min(rev0, rev1, rec0, rec1) <= 0:
        return _fail_insufficient_data(name, "non-positive revenue or receivables")
    rev_cagr = (rev1 / rev0) ** 0.5 - 1
    rec_cagr = (rec1 / rec0) ** 0.5 - 1
    passed = rec_cagr <= rev_cagr * RECEIVABLES_VELOCITY_MULTIPLIER
    return GateResult(name, passed,
                      f"receivables CAGR={rec_cagr:+.1%} vs revenue CAGR={rev_cagr:+.1%}",
                      round(rec_cagr - rev_cagr, 4))


def passes_interest_coverage(*, operating_profit_5y_series: Optional[list[float]],
                             interest_expense_latest: Optional[float],
                             is_financial: bool = False) -> GateResult:
    """Operating profit over interest: catches a debt bomb on the income
    statement before it shows in the balance-sheet D/E. Zero interest is
    debt-free and passes."""
    name = "passes_interest_coverage"
    if is_financial:
        return _financial_na(name, "interest coverage")
    if INTEREST_COVERAGE_MIN is None:
        return _abstain_withheld(name, "coverage floor")
    if not operating_profit_5y_series:
        return _fail_insufficient_data(name, "operating profit absent from the record")
    if interest_expense_latest is None or interest_expense_latest <= 0:
        return GateResult(name, True, "no material interest expense: coverage not a constraint")
    coverage = operating_profit_5y_series[-1] / interest_expense_latest
    return GateResult(name, coverage >= INTEREST_COVERAGE_MIN,
                      f"coverage={coverage:.2f}x", round(coverage, 3))


def passes_peak_margin(*, operating_profit_5y_series: Optional[list[float]],
                       revenue_5y_series: Optional[list[float]],
                       is_financial: bool = False) -> GateResult:
    """A name looks cheap on P/E when earnings sit at a cyclical peak. Compares
    the latest operating margin with the median of the prior years. Operating
    margin, not net margin: a deleveraging story spikes net margin mechanically
    while the operating business has not peaked at all. A loss-making history
    passes (any profitability is improvement, not a peak)."""
    name = "passes_peak_margin"
    if is_financial:
        return _financial_na(name, "peak margin gate")
    if PEAK_MARGIN_MAX_MULTIPLIER is None:
        return _abstain_withheld(name, "peak margin multiple")
    if not operating_profit_5y_series or not revenue_5y_series:
        return _fail_insufficient_data(name, "operating profit or revenue series missing")
    n = min(len(operating_profit_5y_series), len(revenue_5y_series))
    opm = [op / rev for op, rev in zip(operating_profit_5y_series[-n:], revenue_5y_series[-n:])
           if op is not None and rev is not None and rev > 0]
    if len(opm) < 3:
        return _fail_insufficient_data(name, f"only {len(opm)} valid margin points")
    median = statistics.median(opm[:-1])
    if median <= 0:
        return GateResult(name, True, "loss-making margin history: not a peak", round(opm[-1], 4))
    passed = opm[-1] <= median * PEAK_MARGIN_MAX_MULTIPLIER
    return GateResult(name, passed, f"latest OPM={opm[-1]:.1%} vs median {median:.1%}",
                      round(opm[-1] / median, 4))


# --- Orchestration -----------------------------------------------------------

@dataclass
class GateInputs:
    """Everything the stack needs for one stock, assembled by the sweep from
    the fundamentals record, the peer cohort and the price history. The gates
    themselves are pure functions over these inputs."""
    symbol: str
    is_financial: bool = False
    ocf_5y_series: Optional[list[float]] = None
    net_profit_5y_series: Optional[list[float]] = None
    current_de: Optional[float] = None
    de_1y_ago: Optional[float] = None
    de_3y_avg: Optional[float] = None
    net_worth_current: Optional[float] = None
    net_worth_5y_ago: Optional[float] = None
    current_pe: Optional[float] = None
    peer_pe_list: Optional[list[float]] = None
    current_roce: Optional[float] = None
    current_ev_ebitda: Optional[float] = None
    peer_ev_ebitda_list: Optional[list[float]] = None
    months_from_ath: Optional[float] = None
    severity_score: Optional[int] = None
    severity_votes: Optional[list] = None
    promoter_pledge_pct: Optional[float] = None
    revenue_3y_series: Optional[list[float]] = None
    receivables_3y_series: Optional[list[float]] = None
    operating_profit_5y_series: Optional[list[float]] = None
    interest_expense_latest: Optional[float] = None
    revenue_5y_series: Optional[list[float]] = None
    # Peer cohort provenance: the selector's stamp, and the peer symbols in the
    # SAME ORDER as the two multiple lists. Recorded, never a verdict input.
    peer_selection: Optional[dict] = None
    peer_symbols: Optional[list[str]] = None


def evaluate_all_gates(inputs: GateInputs) -> GatesEvaluation:
    """Run the full stack. The order is stable (audit tooling indexes it)."""
    i = inputs
    return GatesEvaluation(symbol=i.symbol, results=[
        passes_earnings_quality(ocf_5y_series=i.ocf_5y_series,
                                net_profit_5y_series=i.net_profit_5y_series,
                                is_financial=i.is_financial),
        passes_debt_trajectory(current_de=i.current_de, de_1y_ago=i.de_1y_ago,
                               de_3y_avg=i.de_3y_avg, is_financial=i.is_financial),
        passes_roiic_proxy(net_profit_5y_series=i.net_profit_5y_series,
                           net_worth_current=i.net_worth_current,
                           net_worth_5y_ago=i.net_worth_5y_ago, is_financial=i.is_financial),
        passes_valuation_sector_relative(current_pe=i.current_pe, peer_pe_list=i.peer_pe_list,
                                         current_roce=i.current_roce,
                                         peer_selection=i.peer_selection,
                                         peer_symbols=i.peer_symbols),
        passes_distress_age(months_from_ath=i.months_from_ath,
                            severity_score=i.severity_score,
                            severity_votes=i.severity_votes),
        passes_promoter_pledge(promoter_pledge_pct=i.promoter_pledge_pct),
        passes_receivables_velocity(revenue_3y_series=i.revenue_3y_series,
                                    receivables_3y_series=i.receivables_3y_series),
        passes_interest_coverage(operating_profit_5y_series=i.operating_profit_5y_series,
                                 interest_expense_latest=i.interest_expense_latest,
                                 is_financial=i.is_financial),
        passes_peak_margin(operating_profit_5y_series=i.operating_profit_5y_series,
                           revenue_5y_series=i.revenue_5y_series,
                           is_financial=i.is_financial),
        passes_valuation_ev_ebitda(current_ev_ebitda=i.current_ev_ebitda,
                                   peer_ev_ebitda_list=i.peer_ev_ebitda_list,
                                   current_roce=i.current_roce,
                                   peer_selection=i.peer_selection,
                                   peer_symbols=i.peer_symbols),
    ])


# --- The family policy -------------------------------------------------------
# It replaced a binary rule (any not-passed gate capped the name at the bottom
# of the ladder), under which a single soft miss, or a field nobody could
# measure, decided a name before the synthesizer could weigh it. That legacy
# rule is retired and not shipped.

GATE_POLICY_VERSION = "v2"

GATE_FAMILY = {
    "passes_earnings_quality": "quality",
    "passes_roiic_proxy": "quality",
    "passes_debt_trajectory": "leverage",
    "passes_interest_coverage": "leverage",
    "passes_valuation_sector_relative": "valuation",
    "passes_valuation_ev_ebitda": "valuation",
    "passes_distress_age": "timing",
    "passes_promoter_pledge": "governance",
    "passes_receivables_velocity": "accounting",
    "passes_peak_margin": "cycle",
}
# The core value pillars. A core pillar nobody could measure blocks Tier A but
# does not sink the name; Cortex is told and weighs it on everything else.
CORE_FOR_A_FAMILIES = frozenset({"quality", "leverage", "valuation"})
# Every member of one of these failing is a structural miss, not a soft one.
HARD_FAMILY_MISS_FAMILIES = frozenset({"quality", "leverage", "valuation"})
# A real FAIL here (not an ABSTAIN) is a hard blocker.
HARD_BLOCKER_FAMILIES = frozenset({"governance", "accounting"})
# Missing data on these is itself the red flag, so it stays a FAIL.
DISCLOSURE_REQUIRED_GATES = frozenset({"passes_promoter_pledge"})
# Below this many measurable core pillars a name is a data-coverage gap, not a
# value candidate: kept at the floor rather than handed over as an empty shell.
MIN_MEASURED_CORE_FAMILIES: Optional[int] = None  # withheld from the public edition
# The most failed families a name may carry and still reach the middle tier.
MAX_FAILED_FAMILIES_FOR_B: Optional[int] = None  # withheld from the public edition

# One value per branch of `_ceiling_and_reason`, in branch order.
CEILING_REASONS = ("hard_blocker", "structural", "withheld", "unmeasurable",
                   "clean", "core_abstain", "soft_miss", "family_misses")
_VALID_STATUSES = ("PASS", "FAIL", "ABSTAIN")


@dataclass(frozen=True)
class GateFailureProfile:
    """The status / family view over one evaluation."""
    statuses: dict[str, str]
    failed_gates: list[str]
    abstained_gates: list[str]
    failed_families: set[str]
    abstained_families: set[str]
    hard_blocker_gates: list[str]
    structural_failed_families: set[str]
    measured_core_families: int


def gate_status(result: GateResult) -> str:
    """PASS / FAIL / ABSTAIN for one gate. A withheld threshold abstains on
    every gate. A data gap abstains except on a disclosure-required gate.
    "Have data, it is bad" branches carry their own `why` and stay FAIL, which
    is what keeps sustained loss-makers out of the upper tiers."""
    if result.passed:
        return "PASS"
    why = result.why or ""
    if why.startswith(WITHHELD_PREFIX):
        return "ABSTAIN"
    if result.name not in DISCLOSURE_REQUIRED_GATES and why.startswith(INSUFFICIENT_PREFIX):
        return "ABSTAIN"
    return "FAIL"


def gate_vector_summary(gates: Optional[dict]) -> dict:
    """Summarize a SERIALIZED gate vector (``GatesEvaluation.to_dict()``, or a
    legacy dict off an old artifact) for operator surfaces only.

    Reads each row's ``status`` exactly as it was PERSISTED and never re-derives
    it from ``why``: re-grading an old vector under today's rules is the drift
    the validation ledger exists to detect. A legacy row with no status falls
    back to ``passed`` -> PASS / FAIL, and ``has_status`` reports False. A
    repeated gate name keeps its first row. Nothing in the analysis path calls
    this; the ceiling is owned by ``gate_failure_profile``."""
    rows = []
    if isinstance(gates, dict) and isinstance(gates.get("results"), list):
        rows = [r for r in gates["results"] if isinstance(r, dict)]
    statuses: dict[str, str] = {}
    passed, failed, abstained = [], [], []
    n_with_status = 0
    for r in rows:
        name = r.get("name")
        if not isinstance(name, str) or not name or name in statuses:
            continue
        raw = r.get("status")
        if isinstance(raw, str) and raw.strip().upper() in _VALID_STATUSES:
            status = raw.strip().upper()
            n_with_status += 1
        else:
            status = "PASS" if r.get("passed") else "FAIL"
        statuses[name] = status
        {"PASS": passed, "ABSTAIN": abstained}.get(status, failed).append(name)
    n = len(statuses)
    return {"has_status": n > 0 and n_with_status == n, "statuses": statuses,
            "passed": passed, "failed": failed, "abstained": abstained,
            "n_pass": len(passed), "n_fail": len(failed), "n_abstain": len(abstained),
            "n": n}


def _build_profile(statuses: dict, order: list) -> GateFailureProfile:
    failed_families, abstained_families, hard_blockers = set(), set(), []
    for name, status in statuses.items():
        fam = GATE_FAMILY.get(name)
        if fam is None:
            continue
        if status == "FAIL":
            failed_families.add(fam)
            if fam in HARD_BLOCKER_FAMILIES:
                hard_blockers.append(name)
        elif status == "ABSTAIN":
            abstained_families.add(fam)
    structural = set()
    for fam in HARD_FAMILY_MISS_FAMILIES:
        members = [n for n, f in GATE_FAMILY.items() if f == fam]
        if members and all(statuses.get(m) == "FAIL" for m in members):
            structural.add(fam)
    measured = sum(
        1 for fam in CORE_FOR_A_FAMILIES
        if any(statuses.get(m) in ("PASS", "FAIL") for m, f in GATE_FAMILY.items() if f == fam))
    return GateFailureProfile(
        statuses=statuses,
        failed_gates=[n for n in order if statuses.get(n) == "FAIL"],
        abstained_gates=[n for n in order if statuses.get(n) == "ABSTAIN"],
        failed_families=failed_families, abstained_families=abstained_families,
        hard_blocker_gates=hard_blockers, structural_failed_families=structural,
        measured_core_families=measured)


def gate_failure_profile(evaluation: GatesEvaluation) -> GateFailureProfile:
    """The single source of truth for the ceiling and the serialized audit
    fields. ABSTAIN gates are not counted as failures."""
    return _build_profile({r.name: gate_status(r) for r in evaluation.results},
                          [r.name for r in evaluation.results])


def _ceiling_and_reason(profile: GateFailureProfile) -> tuple[str, str]:
    """(ceiling letter, reason) over a profile. ONE branch walk, so the letter
    and its stamped reason can never drift apart."""
    if profile.hard_blocker_gates:
        return "C", "hard_blocker"
    if profile.structural_failed_families:
        return "C", "structural"
    if MIN_MEASURED_CORE_FAMILIES is None or MAX_FAILED_FAMILIES_FOR_B is None:
        # The family counts that separate the tiers are withheld from the
        # public edition: the walk stops at the floor and says so.
        return "C", "withheld"
    if profile.measured_core_families < MIN_MEASURED_CORE_FAMILIES:
        return "C", "unmeasurable"
    if not profile.failed_families and not (profile.abstained_families & CORE_FOR_A_FAMILIES):
        return "A", "clean"
    if len(profile.failed_families) <= MAX_FAILED_FAMILIES_FOR_B:
        return ("B", "soft_miss") if profile.failed_families else ("B", "core_abstain")
    return "C", "family_misses"


def tier_ceiling_from_gates(evaluation: GatesEvaluation) -> str:
    """The MAXIMUM tier Cortex may assign this name. The synthesizer may assign
    the ceiling or anything below it, never above. The ceiling is not the
    assigned tier.

      A  nothing failed and every core pillar was measured.
      B  few enough failed families (the count is withheld), no hard blocker,
         no structural miss: a soft miss or an unmeasured core pillar lands
         here.
      C  a hard blocker FAILed, a core family failed outright, too few core
         pillars were measurable, or several independent families failed.

    The lower tiers belong to the model and the severity floor, not to this
    function. In the public edition every gate abstains on its withheld
    threshold, so a non-financial name reads "unmeasurable" here."""
    return _ceiling_and_reason(gate_failure_profile(evaluation))[0]


# --- Record staleness (measured, never applied) ------------------------------
# The gates that grade off ANNUAL statement rows. A record whose newest annual
# year is behind the year a current record should carry grades them on old
# numbers; this names them. It moves no `passed`, no status and no ceiling.

ANNUAL_RECORD_GATES = frozenset({
    "passes_earnings_quality", "passes_roiic_proxy", "passes_debt_trajectory",
    "passes_interest_coverage", "passes_peak_margin", "passes_receivables_velocity",
})
_YEAR_RE = re.compile(r"(\d{4})")


def latest_annual_year(series: Optional[dict]) -> Optional[int]:
    """Newest 4-digit year label carrying a numeric value in a year-keyed
    statement table, or None when the table is absent or unreadable."""
    if not isinstance(series, dict):
        return None
    years = []
    for label, value in series.items():
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        m = _YEAR_RE.search(str(label))
        if m:
            years.append(int(m.group(1)))
    return max(years) if years else None


def stale_record_gates(gates: Optional[dict], *, annual_series: Optional[dict] = None,
                       as_of: Optional[date] = None) -> dict:
    """The annual gates a stale record graded, with the basis. A record with no
    readable annual year is stale (fail-closed), never a silent pass."""
    # One shared definition of the annual year a current record should carry.
    from data.fundamentals_series import expected_annual_year  # not published in the public edition
    expected = expected_annual_year(as_of or date.today())
    latest = latest_annual_year(annual_series)
    stale = latest is None or latest < expected
    graded = [r.get("name") for r in (gates or {}).get("results") or [] if isinstance(r, dict)]
    return {"stale_record": [n for n in graded if n in ANNUAL_RECORD_GATES] if stale else [],
            "latest_year": latest, "expected_year": expected,
            "basis": "no_annual_series" if latest is None else "annual_headers"}
