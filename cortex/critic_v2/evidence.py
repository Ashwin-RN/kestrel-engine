"""The council critic's citation checker: code checks every fact a critique cites.

Under the second council contract a critic's finding cannot just assert
things. A critic first reads the evidence blind, then critiques the author's
draft; the facts cited in the critique pass are the ones checked here. Each
finding carries `facts: [{field_path, value}]`: a dotted path into the dossier
the critic read for that name, and the value it quoted from there.
This module resolves each path and compares the quote with the field in CODE,
then labels the fact:

  CONFIRMED    the path resolves and the quoted value matches the field.
  MISMATCH     the path resolves but the quote does not match: a wrong number,
               a paraphrase that changes a figure, a null field quoted as a
               value. An unscoreable quote is a MISMATCH, never UNCHECKABLE,
               so it cannot leave the denominator of the pass rate.
  NOT_FOUND    the path does not resolve on the dossier the critic was shown:
               an invented field, one the model's copy never carried, or a
               lookup id never returned for that name.
  UNCHECKABLE  a scalar quote against a list or object that matches no
               element, or a field type the comparator does not handle. It
               votes neither way.

What is checked is the CITED VALUE, never the reasoning: the finding's
inference, its confidence and whether the facts support its action are for the
arbiter to rule on. The checker labels; it rejects nothing. Downstream (in
modules not published here), a finding with any MISMATCH or NOT_FOUND fact
reaches the arbiter marked `evidence_ok: false`, an UNCHECKABLE fact does not
count against it, and only CONFIRMED citations count toward the rule that
binds a point both critic seats raised on one field.

"The dossier the critic read" is rendered through the one payload seam the
critic's input is built from, so a field dropped from the model's copy comes
back NOT_FOUND, the correct verdict. Lookup results the critic was shown
(stored filing text, looked up on its request; nothing is fetched live) ride
on it under `lookup.<id>`.

Numbers are compared against the field's UNIT. Scale forms are accepted only
among fraction, percent and percentage points (a drawdown quoted as a
fraction or as a percent is the same drawdown), never across ratio, rupees,
crore, count, months or score. A quote carrying several numbers confirms only
when every one is an accepted form of the field. Text confirms on normalised
equality, or on a near match in which every number of the quote must appear in
the field, so word overlap never confirms a changed figure. Enum and date fields take equality only.

Public edition: an excerpt. Every tolerance, the near-match token rules and
the lookup span floor are constants set to None; with one withheld the checker
takes the strictest reading and says so in `detail` (numbers match exactly,
text must be equal, a lookup fact cannot confirm). The full unit table (a
sample stands in), the batch renderer, the table-coverage resolver, the
gate-recital scorer and the table's prompt writer are not published.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

CONFIRMED = "CONFIRMED"
MISMATCH = "MISMATCH"
NOT_FOUND = "NOT_FOUND"
UNCHECKABLE = "UNCHECKABLE"
STATUSES = (CONFIRMED, MISMATCH, NOT_FOUND, UNCHECKABLE)

UNITS = ("fraction", "percent", "pp", "ratio", "inr", "crore", "months", "count",
         "score", "enum", "bool", "text", "date")

# --- Tolerances (withheld) ---------------------------------------------------
# Relative tolerance on |actual|; absolute tolerance on the display scale; how
# near a quote rounded to its own decimals must still sit; the text near-match
# (minimum quote tokens, share that must appear); the lookup span floor (words).
REL_TOL: Optional[float] = None  # withheld from the public edition
DISPLAY_ABS_TOL: Optional[float] = None  # withheld from the public edition
ROUNDING_REL_CAP: Optional[float] = None  # withheld from the public edition
TEXT_MIN_TOKENS: Optional[int] = None  # withheld from the public edition
TEXT_TOKEN_SHARE: Optional[float] = None  # withheld from the public edition
LOOKUP_MIN_TOKENS: Optional[int] = None  # withheld from the public edition

WITHHELD = "withheld from the public edition"
_EPS = 1e-12     # float noise, not a tolerance
_ELEMENT_KEYS = ("name", "metric", "symbol")   # how a list element is named in a path
LOOKUP_ROOT = "lookup"

# --- The unit table (a sample; the full table is not published) --------------
# `[*]` = any list index; `[name=x]` = a named gate result, tried BEFORE `[*]`.
FIELD_UNITS: dict[str, str] = {
    "tier_ceiling": "enum",
    "months_from_ath": "months",
    "market.drawdown_pct": "fraction",
    "market.relative_drawdown.cohort_delta_pp": "pp",
    "diagnosis.severity_score": "score",
    "diagnosis.red_flags[*]": "text",
    "fundamentals.roce": "percent",
    "fundamentals.debt_to_equity": "ratio",
    "fundamentals.market_cap_cr": "crore",
    "gate_results.all_passed": "bool",
    "gate_results.results[*].numeric_value": "ratio",
    "gate_results.results[name=passes_distress_age].numeric_value": "months",
}


def render_dossier(dossier: dict) -> dict:
    """What the model saw for one dossier, through the one payload seam."""
    from cortex.synth_payload import render_llm_payload  # not published in the public edition
    return json.loads(render_llm_payload({"DOSSIERS": [dossier]}))["DOSSIERS"][0]


# --- Locating a cited field ----------------------------------------------------

_MISSING = object()


def normalize_path(path: str) -> str:
    """Stripped, a leading `dossier.` dropped. The one normalisation, shared
    with the code that applies rulings, so a path the checker confirms is
    never read there as a different field."""
    p = str(path or "").strip()
    return p[len("dossier."):] if p.startswith("dossier.") else p


def _tokens(path: str) -> Optional[list[tuple]]:
    """("key", k) / ("idx", i) / ("sel", k, v) / ("elem", x) tokens, or None
    when malformed: `a.b[2]`, `results[name=x]`, `wins[roce]`. A wildcard is
    never a valid citation. (Condensed from a character scanner.)"""
    p = normalize_path(path)
    if p.count("[") != p.count("]"):
        return None
    out: list[tuple] = []
    for m in re.finditer(r"\[([^\]]*)\]|([^.\[\]]+)", p):
        if m.group(2) is not None:
            if m.group(2).strip():
                out.append(("key", m.group(2).strip()))
            continue
        inner = m.group(1).strip().strip("'\"")
        if re.fullmatch(r"\d+", inner):
            out.append(("idx", int(inner)))
        elif inner in ("*", ""):
            return None
        elif "=" in inner:
            k, v = inner.split("=", 1)
            out.append(("sel", k.strip(), v.strip().strip("'\"")))
        else:
            out.append(("elem", inner))
    return out or None


def _match_elem(lst: list, key: Optional[str], val: str):
    """First dict in `lst` whose `key` (or, key None, whose name / metric /
    symbol) equals `val`, case-insensitively: (index, element)."""
    want = val.strip().casefold()
    for k in ((key,) if key else _ELEMENT_KEYS):
        for i, el in enumerate(lst):
            if isinstance(el, dict) and k in el and str(el[k]).strip().casefold() == want:
                return i, el
    return None, _MISSING


def _walk(rendered: Any, path: str):
    """(value or _MISSING, steps). A step is ("key", k), or ("item", element)
    for a list step, the element kept for the unit lookup."""
    toks = _tokens(path) if isinstance(path, str) else None
    if not toks:
        return _MISSING, []
    node, steps = rendered, []
    for t in toks:
        if t[0] == "key" and isinstance(node, dict):
            if t[1] not in node:
                return _MISSING, steps
            node = node[t[1]]
            steps.append(("key", t[1]))
            continue
        if not isinstance(node, list):
            return _MISSING, steps
        if t[0] == "idx" or (t[0] == "key" and t[1].isdigit()):
            i = int(t[1])
            node = node[i] if i < len(node) else _MISSING
        else:   # `[k=v]`, a bare `[x]`, or a dotted key applied to a list
            _, node = _match_elem(node, t[1] if t[0] == "sel" else None, t[-1])
        if node is _MISSING:
            return _MISSING, steps
        steps.append(("item", node))
    return node, steps


def resolve(rendered: dict, path: str) -> tuple[bool, Any]:
    """(found, value). A key present with a null value is FOUND with None."""
    val, _ = _walk(rendered, path)
    return (False, None) if val is _MISSING else (True, val)


def unit_for(rendered: dict, path: str) -> Optional[str]:
    """The field's unit from the table, the named-gate pattern tried first;
    None when the table does not cover it (still compared, on its raw scale).
    (A peer win or loss value takes its sibling metric's unit; not shown.)"""
    val, steps = _walk(rendered, path)
    if val is _MISSING:
        return None
    named, generic = "", ""
    for kind, v in steps:
        sep = "." if generic and kind == "key" else ""
        if kind == "key":
            named, generic = named + sep + v, generic + sep + v
        else:
            nm = v.get("name") if isinstance(v, dict) else None
            named += f"[name={nm}]" if isinstance(nm, str) else "[*]"
            generic += "[*]"
    cands = [named, generic] + ([named + "[*]", generic + "[*]"] if isinstance(val, list) else [])
    for p in cands:                   # a whole list cited takes its elements' unit
        if p in FIELD_UNITS:
            return FIELD_UNITS[p]
    return None


# --- Comparing the quote with the field ------------------------------------------

# A number glued to a preceding letter (a fiscal-year or quarter label) is a
# label, not a value.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _norm(s) -> str:
    return " ".join(str(s).casefold().split()).strip(" .,;:'\"")


def _toks(s) -> list[str]:
    return _TOKEN_RE.findall(str(s).casefold())


def _quote_numbers(q) -> list[tuple[float, int, bool]]:
    """[(value, decimals, followed by a % sign)] parsed from a quote; commas
    and currency or multiplier words around a number are tolerated."""
    if _is_num(q):
        s = repr(float(q)) if isinstance(q, float) else str(q)
        dec = 0 if isinstance(q, int) or "e" in s else len(s.split(".")[1].rstrip("0"))
        return [(float(q), dec, False)]
    s, out = str(q), []
    for m in _NUM_RE.finditer(s):
        txt = m.group(0).replace(",", "").replace("+", "")
        try:
            v = float(txt)
        except ValueError:
            continue
        dec = len(txt.split(".")[1]) if "." in txt else 0
        out.append((v, dec, s[m.end():].lstrip().startswith("%")))
    return out


def _close(a: float, q: float, dec: int, abs_tol: Optional[float]) -> bool:
    """Within tolerance, or the actual rounded to the quote's own decimals and
    still near it (so a quote of 0 never confirms a small non-zero field)."""
    diff = abs(q - a)
    if REL_TOL is None or abs_tol is None:
        return diff <= _EPS        # tolerances withheld: exact match only
    if diff <= max(REL_TOL * abs(a), abs_tol) + _EPS:
        return True
    return ROUNDING_REL_CAP is not None and a != 0 and diff <= ROUNDING_REL_CAP * abs(a) \
        and abs(round(a, dec) - q) <= _EPS


def _num_ok(a: float, q: float, dec: int, unit: Optional[str], pct_sign: bool,
            drawdown: bool) -> bool:
    """Is `q` an accepted form of `a` under the field's unit?"""
    tol = DISPLAY_ABS_TOL
    frac_tol = None if tol is None else tol / 100      # the same tolerance, fraction form
    forms: list[tuple[float, Optional[float]]] = []
    if unit == "fraction":
        if not pct_sign:              # "0.66%" never confirms a 66% drawdown
            forms.append((a, frac_tol))
        forms.append((a * 100, tol))
    elif unit in ("percent", "pp"):
        forms.append((a, tol))
        if not pct_sign:
            forms.append((a / 100, frac_tol))
    else:                             # no scale forms across any other unit
        forms.append((a, tol))
    # Critics write a drawdown as negative; elsewhere the sign carries meaning.
    qs = [q, -q] if (drawdown and unit == "fraction") else [q]
    return any(_close(fa, qq, dec, t) for fa, t in forms for qq in qs)


def _text_ok(actual: str, quote: str, unit: Optional[str]) -> tuple[bool, str]:
    a, q = _norm(actual), _norm(quote)
    if a == q:
        return True, "equal"
    if unit in ("enum", "date"):
        return False, "enum/date fields need equality"
    if TEXT_MIN_TOKENS is None or TEXT_TOKEN_SHARE is None:
        return False, f"not equal; the near-match rule is {WITHHELD}"
    qt, at = _toks(quote), set(_toks(actual))
    if len(qt) < TEXT_MIN_TOKENS:
        return False, "too short for a near match, and not equal"
    if q in a:
        return True, "substring"
    missing = {t for t in qt if re.fullmatch(r"\d+(?:\.\d+)?", t) and t not in at}
    if missing:                       # a changed figure never confirms on overlap
        return False, f"numbers {sorted(missing)} not in the field"
    share = sum(1 for t in qt if t in at) / len(qt)
    return (True, "token share") if share >= TEXT_TOKEN_SHARE else (False, "token share too low")


def _compare(actual: Any, quote: Any, unit: Optional[str], path: str) -> tuple[str, str]:
    """(status, detail) for one quote against one resolved field. (The bool
    and null readers are condensed; the full forms take a few more spellings.)"""
    if quote is None or (actual is None and _norm(quote) in ("null", "none", "n/a")):
        return (CONFIRMED, "null quoted, null field") if actual is None \
            else (MISMATCH, "null quoted for a non-null field")
    if actual is None:
        return MISMATCH, "the field is null"
    if isinstance(actual, bool):
        b = quote if isinstance(quote, bool) else {"true": True, "false": False}.get(_norm(quote))
        if b is None:
            return MISMATCH, "a bool field needs true/false"
        return (CONFIRMED, "bool") if b == actual else (MISMATCH, "bool differs")
    if _is_num(actual):
        if isinstance(quote, (list, dict, bool)):
            return MISMATCH, "a number field quoted as a container or bool"
        nums = _quote_numbers(quote)
        if not nums:
            return MISMATCH, "the quote carries no number"
        dd = "drawdown" in path.casefold()
        ok = all(_num_ok(float(actual), v, d, unit, pct, dd) for v, d, pct in nums)
        exact = "" if REL_TOL is not None and DISPLAY_ABS_TOL is not None \
            else f"; tolerances {WITHHELD}, exact match only"
        return (CONFIRMED if ok else MISMATCH), \
            f"number {'matches' if ok else 'differs'} ({unit or 'unit unknown'}){exact}"
    if isinstance(actual, str):
        if isinstance(quote, (list, dict)):
            return MISMATCH, "a text field quoted as a container"
        ok, why = _text_ok(actual, json.dumps(quote) if isinstance(quote, bool) else str(quote), unit)
        return (CONFIRMED if ok else MISMATCH), why
    if isinstance(actual, list):
        if isinstance(quote, list):
            if all(_any_elem(actual, q, unit, path) for q in quote):
                return CONFIRMED, "every quoted element found"
            return MISMATCH, "a quoted element is not in the list"
        if _is_num(quote) and float(quote).is_integer() and int(quote) == len(actual):
            return CONFIRMED, "matched the list length"     # "3 red flags"
        if _any_elem(actual, quote, unit, path):
            return CONFIRMED, "matched an element"
        return UNCHECKABLE, "quoted against a list, no element matches"
    # (An object field follows the same shape: a quoted object confirms when
    # every quoted key matches, a scalar when it matches one of the object's
    # values, else UNCHECKABLE. Not shown.)
    return UNCHECKABLE, f"unhandled field type {type(actual).__name__}"


def _any_elem(elems: list, quote: Any, unit: Optional[str], path: str) -> bool:
    for el in elems:
        if isinstance(el, dict) and not isinstance(quote, dict):
            if any(_compare(v, quote, None, path)[0] == CONFIRMED
                   for v in el.values() if not isinstance(v, (dict, list))):
                return True
        elif _compare(el, quote, None if isinstance(el, (dict, list)) else unit,
                      path)[0] == CONFIRMED:
            return True
    return False


def _check_lookup(rendered: dict, field_path: str, quoted: Any) -> dict:
    """A lookup result is kilobytes of filing text, so the near-match rule of
    the short dossier fields would pass a paraphrase: a lookup fact must quote
    a VERBATIM span of the result, of at least LOOKUP_MIN_TOKENS words."""
    rid = field_path[len(LOOKUP_ROOT) + 1:].strip()
    text = (rendered.get(LOOKUP_ROOT) or {}).get(rid)
    base = {"actual": None, "unit": "lookup"}    # never echo kilobytes back
    if not isinstance(text, str):
        return {"status": NOT_FOUND, **base, "detail": f"no lookup result {rid!r} for this name"}
    if LOOKUP_MIN_TOKENS is None:
        return {"status": MISMATCH, **base,
                "detail": f"the span floor is {WITHHELD}; a lookup fact cannot confirm"}
    if not isinstance(quoted, str) or len(_toks(quoted)) < LOOKUP_MIN_TOKENS:
        return {"status": MISMATCH, **base, "detail": "a lookup fact quotes a longer verbatim span"}
    if _norm(quoted) in _norm(text):
        return {"status": CONFIRMED, **base, "detail": "verbatim span of the lookup"}
    return {"status": MISMATCH, **base, "detail": "not a verbatim span of the lookup"}


# --- The public API ------------------------------------------------------------

def check_fact(rendered: dict, field_path: str, quoted: Any) -> dict:
    """{status, actual, unit, detail} for one cited fact on one RENDERED dossier."""
    if not isinstance(field_path, str) or not field_path.strip():
        return {"status": NOT_FOUND, "actual": None, "unit": None, "detail": "no field_path"}
    if field_path.startswith(LOOKUP_ROOT + "."):
        return _check_lookup(rendered, field_path, quoted)
    found, val = resolve(rendered, field_path)
    if not found:
        return {"status": NOT_FOUND, "actual": None, "unit": None,
                "detail": "path does not resolve on the rendered dossier"}
    unit = unit_for(rendered, field_path)
    status, detail = _compare(val, quoted, unit, field_path)
    return {"status": status, "actual": val, "unit": unit, "detail": detail}


def check_row_facts(rows: dict, rendered_by_symbol: dict) -> dict:
    """Every fact of every finding across one critic pass's validated rows
    ({SYM: row}), each checked against that name's rendered dossier (a name
    with no rendered dossier makes its facts NOT_FOUND). `confirmed_rate` =
    confirmed / (n_facts - uncheckable), None when that denominator is 0: the
    rate a critic seat is scored on. (A coverage count of the paths a clean
    row says it looked at is not shown.)"""
    by_sym = {str(k).upper(): v for k, v in (rendered_by_symbol or {}).items()}
    per_fact: list[dict] = []
    for sym, row in (rows or {}).items():
        if not isinstance(row, dict):
            continue
        s = str(row.get("symbol") or sym).upper()
        rendered = by_sym.get(s)
        findings = row.get("findings") if isinstance(row.get("findings"), list) else []
        for i, f in enumerate(findings):
            facts = f.get("facts") if isinstance(f, dict) else None
            for j, fa in enumerate(facts if isinstance(facts, list) else []):
                if not isinstance(fa, dict):
                    continue
                path, quoted = fa.get("field_path"), fa.get("value")
                res = check_fact(rendered, path, quoted) if rendered is not None else \
                    {"status": NOT_FOUND, "actual": None, "unit": None,
                     "detail": "no rendered dossier for this name"}
                per_fact.append({"symbol": s, "finding": i, "fact": j,
                                 "field_path": path, "quoted": quoted, **res})
    counts = {st: sum(1 for x in per_fact if x["status"] == st) for st in STATUSES}
    n = len(per_fact)
    denom = n - counts[UNCHECKABLE]
    return {"n_facts": n, "confirmed": counts[CONFIRMED], "mismatch": counts[MISMATCH],
            "not_found": counts[NOT_FOUND], "uncheckable": counts[UNCHECKABLE],
            "confirmed_rate": (counts[CONFIRMED] / denom) if denom else None,
            "per_fact": per_fact}
