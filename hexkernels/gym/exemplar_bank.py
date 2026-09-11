"""Retrieve a genuine-mechanism exemplar kernel to inject into a repair turn.

The bank (built offline by ``build_exemplar_bank`` from TRAIN-split ``expert.c``
kernels, verified correct AND genuine per mechanism) maps each mechanism to a list
of reference kernels. At repair time, when the gym's bottleneck analysis reports a
MISSING mechanism, :func:`retrieve` returns the closest genuine exemplar for that
mechanism so the model has a concrete, reachable pattern to imitate.

Pure and offline: the only I/O is :func:`load_bank` (reads the prebuilt JSON).
Ranking is deterministic (dtype match > family match > prompt token overlap) so a
run is reproducible. Leak-guard: an exemplar with the same ``task_id`` as the target
is never returned (the bank is train-only, but this is belt-and-suspenders)."""
import json
import re

_TOK = re.compile(r"[a-z0-9]+")

# Scoring weights: an exact dtype match is the strongest signal that a pattern
# transfers (i8 DMA staging differs from fp16), then same op-family, then the
# continuous prompt-overlap tiebreak in [0,1].
_W_DTYPE = 3.0
_W_FAMILY = 2.0


def load_bank(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(s):
    return set(_TOK.findall((s or "").lower()))


def _family(task_id):
    return (task_id or "").split("_")[0]


def _score(entry, dtype, family, ptoks):
    s = 0.0
    if dtype and entry.get("dtype") == dtype:
        s += _W_DTYPE
    if family and entry.get("family") == family:
        s += _W_FAMILY
    etoks = _tokens(entry.get("prompt"))
    if etoks and ptoks:
        s += len(etoks & ptoks) / len(etoks | ptoks)   # Jaccard, 0..1
    return s


def retrieve(bank, mechanism, spec, task_id, prompt):
    """Best genuine exemplar for `mechanism`, or None if the bank has none.

    Ranks bank[mechanism] by (dtype match, family match, prompt overlap) and
    returns the top entry (a dict with at least ``task_id`` and ``code``),
    excluding any entry whose ``task_id`` equals the target (leak-guard)."""
    cands = (bank or {}).get(mechanism) or []
    dtype = (spec or {}).get("dtype")
    fam = _family(task_id)
    ptoks = _tokens(prompt)
    best, best_score = None, float("-inf")
    for e in cands:
        if e.get("task_id") == task_id:
            continue
        sc = _score(e, dtype, fam, ptoks)
        if sc > best_score:
            best, best_score = e, sc
    return best


def render_exemplar(entry, mechanism):
    """Repair-turn injection block for a retrieved exemplar. Empty if None."""
    if not entry:
        return ""
    return (f"\n\nWorked example -- a GENUINE {mechanism} kernel from a related task "
            f"({entry.get('task_id')}). Study how it reaches the {mechanism} mechanism "
            f"and adapt that pattern to YOUR task (do not copy it verbatim; your shapes "
            f"and dtypes differ):\n```c\n" + (entry.get("code") or "") + "\n```")
