"""Undeclared-intrinsic repair for the gym's multiturn loop (NO training).

Empirically, 100% of the 30B's holdout compile failures are HALLUCINATED HVX
intrinsic names -- ``call to undeclared function 'Q6_...'`` where the name is not in
the Hexagon ISA (e.g. ``Q6_Vuw_vmax_VuwVuw`` -> real is ``Q6_Vuh_vmax_VuhVuh``). Every
such name has a close REAL neighbour in ``hvx_hexagon_protos.h`` (938 intrinsics). This
module parses the SDK header for the valid name set, and turns a compiler error into a
targeted repair hint (nearest real intrinsics) that is appended to the turn's feedback so
the model fixes the name instead of re-hallucinating. Timing-independent, purely static.
"""
import difflib
import glob
import os
import re

_UNDECLARED = re.compile(r"undeclared function '(Q6_[A-Za-z0-9_]+)'")
_NAME = re.compile(r"\bQ6_[A-Za-z0-9_]+\b")
_CACHE = {}


def _header_paths(sdk_root):
    """The HVX + scalar intrinsic prototype headers under an installed SDK."""
    pats = ["tools/HEXAGON_Tools/*/Tools/lib/clang/*/include/hvx_hexagon_protos.h",
            "tools/HEXAGON_Tools/*/Tools/lib/clang/*/include/hexagon_protos.h"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(sdk_root, p))
    return out


def valid_intrinsics(sdk_root):
    """Set of every real ``Q6_*`` intrinsic name declared in the SDK headers (cached)."""
    key = os.path.abspath(sdk_root)
    if key in _CACHE:
        return _CACHE[key]
    names = set()
    for h in _header_paths(sdk_root):
        try:
            with open(h, encoding="utf-8", errors="ignore") as f:
                names.update(_NAME.findall(f.read()))
        except OSError:
            continue
    _CACHE[key] = names
    return names


def undeclared_names(error_text):
    """The Q6_ intrinsic names the compiler flagged as undeclared (deduped, ordered)."""
    seen, out = set(), []
    for n in _UNDECLARED.findall(error_text or ""):
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def suggest(error_text, valid, n=3, cutoff=0.6):
    """{bad_name: [nearest real intrinsics]} for each undeclared name in the error."""
    out = {}
    for bad in undeclared_names(error_text):
        out[bad] = difflib.get_close_matches(bad, valid, n=n, cutoff=cutoff)
    return out


def repair_hint(error_text, valid):
    """A feedback line naming each hallucinated intrinsic + its nearest REAL neighbours,
    or "" if the error has no undeclared-Q6 intrinsic. Appended to the turn observation."""
    sug = suggest(error_text, valid)
    if not sug:
        return ""
    lines = ["COMPILE FIX -- these are NOT real HVX intrinsics; replace each with the "
             "nearest valid one (mind operand type/signedness):"]
    for bad, near in sug.items():
        lines.append(f"  {bad}  ->  {', '.join(near) if near else '(no close match; pick a real Q6_ intrinsic)'}")
    return "\n".join(lines)
