"""Which HVX intrinsics does the BACKEND actually implement for this target?

WHY THIS EXISTS -- IT COST A WHOLE BATCH OF CANDIDATES
-----------------------------------------------------
`hvx_hexagon_protos.h` DECLARES intrinsics the compiler cannot generate code for.
`Q6_Vsf_vmpy_VsfVsf`, `Q6_Vhf_vmpy_VhfVhf` and `Q6_Vhf_vcvt_VsfVsf` are all in the
header, all accept the right argument types, and all pass `-fsyntax-only` --
because syntax checking stops before instruction selection. They then fail at
`-c` with:

    fatal error: error in backend: Cannot select: intrinsic
                 %llvm.hexagon.V6.vmpy.sf.sf.128B

Four of batch 4's five candidates were written against those intrinsics and every
one of them failed that way, after the reference simulations had already been paid
for. The header is a declaration, not a capability list, and `-fsyntax-only` is not
a compile.

So: probe it. Each intrinsic is compiled ALONE, to an object file, and reported
selectable or not. The result is a fact rather than a reading of a header.

    python -m hexkernels.forge.isa_probe --family float
    python -m hexkernels.forge.isa_probe --select Q6_Vqf32_vmpy_VsfVsf,Q6_Vsf_vmpy_VsfVsf

WHAT THIS IS NOT
----------------
Selectable does not mean correct, fast, or semantically what you assumed. It means
the compiler will emit an instruction for it. Correctness is still the golden
harness's job.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

from hexkernels.core import target as _target
from hexkernels.core.toolchain import (COMPILER, CXX_STD, DEFAULT_SDK_ROOT, _exe,
                                    find_toolchain_bin, toolchain_env)

HEADER_REL = os.path.join("..", "target", "hexagon", "include",
                          "hvx_hexagon_protos.h")

# One representative operand expression per HVX type token, so a probe body can be
# generated from the intrinsic's NAME. Keyed by the type letters the vendor uses
# in the `_V<t>` suffixes.
_OPERAND = {
    "V": "v", "Vb": "v", "Vub": "v", "Vh": "v", "Vuh": "v", "Vw": "v",
    "Vuw": "v", "Vhf": "v", "Vsf": "v", "Vqf16": "v", "Vqf32": "v",
    "Vbf": "v", "W": "w", "Wsf": "w", "Wqf32": "w", "Whf": "w",
    "R": "r", "Rt": "r", "Rhf": "r", "Rsf": "r", "A": "m",
    "Q": "q",
}

# The result is RETURNED, and that detail is the whole probe.
#
# The first version of this stored into a `static HVX_Vector sink`. Internal
# linkage, never read, so the store was dead, so the intrinsic was deleted before
# instruction selection ever saw it -- and every one of 27 intrinsics came back
# "selectable", including the two that had just failed in a real kernel. The
# object file was `jumpr r31` and nothing else. An assertion that cannot fail is
# worse than no assertion, so `probe_one` now also DISASSEMBLES the object and
# reports `vacuous` if the body optimised away.
_BODY = r"""
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
extern "C" HVX_Vector probe(HVX_Vector v, HVX_VectorPair w, HVX_VectorPred q,
                            int r, void *m) {
    (void)w; (void)q; (void)r; (void)m;
    return (HVX_Vector)(%CALL%);
}
"""

# A separate body for the intrinsics that RETURN A VECTOR PAIR. Casting a pair to
# HVX_Vector is a compile error, which the first pass reported as "not probed" --
# so `Q6_Wqf32_vmpy_VhfVhf` and `Q6_Wsf_vcvt_Vhf`, the only widening paths from
# fp16 to 32-bit float, were the two intrinsics whose status mattered most and the
# two the prober could not answer for.
_BODY_PAIR = r"""
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
extern "C" HVX_VectorPair probe(HVX_Vector v, HVX_VectorPair w, HVX_VectorPred q,
                                int r, void *m) {
    (void)w; (void)q; (void)r; (void)m;
    return (HVX_VectorPair)(%CALL%);
}
"""


def _returns_pair(name: str) -> bool:
    """True when the intrinsic's RESULT is a vector pair, read off its name.

    The vendor spells the result type first: `Q6_W...` returns a pair, `Q6_V...`
    returns a single vector.
    """
    return name.startswith("Q6_W")


def header_path(sdk_root=DEFAULT_SDK_ROOT) -> str:
    return os.path.normpath(os.path.join(find_toolchain_bin(sdk_root), HEADER_REL))


def declared(pattern=None, sdk_root=DEFAULT_SDK_ROOT) -> list:
    """Every `Q6_*` intrinsic name the header declares, optionally filtered."""
    with open(header_path(sdk_root), encoding="utf-8", errors="replace") as f:
        text = f.read()
    names = sorted(set(re.findall(r"\b(Q6_[A-Za-z0-9_]+)\s*\(", text)))
    if pattern:
        rx = re.compile(pattern)
        names = [n for n in names if rx.search(n)]
    return names


def _args_for(name: str) -> str:
    """Argument list for `name`, inferred from its `_V.../R/A` suffix chain.

    `Q6_Vqf32_vmpy_VsfVsf` -> two vector arguments; `Q6_Vh_vsplat_R` -> one scalar.
    Returns None when the suffix is not one this module knows how to feed, which is
    reported as `skipped` rather than guessed at -- a wrong argument list would
    report a perfectly good intrinsic as broken.
    """
    parts = name.split("_")
    if len(parts) < 3:
        return None
    tail = parts[-1]
    toks, i = [], 0
    while i < len(tail):
        for width in (5, 4, 3, 2, 1):          # Vqf32, Vqf16, Vub, Vhf, Vh, V, R, A, W, Q
            tok = tail[i:i + width]
            if tok in _OPERAND:
                toks.append(tok)
                i += width
                break
        else:
            return None
    return ", ".join(_OPERAND[t] for t in toks)


def probe_one(name: str, cc, env, tgt, workdir) -> tuple:
    """(status, detail). One of 'selectable', 'unselectable', 'vacuous', 'skipped'.

    `vacuous` exists because it HAPPENED: if the probe body optimises away, the
    compile succeeds without the intrinsic ever reaching instruction selection and
    a clean exit code means nothing. So the object is disassembled and must contain
    real work.
    """
    args = _args_for(name)
    if args is None:
        return "skipped", "argument list not inferable from the name"
    body = _BODY_PAIR if _returns_pair(name) else _BODY
    src = body.replace("%CALL%", f"{name}({args})")
    path = os.path.join(workdir, "probe.cpp")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    obj = os.path.join(workdir, "probe.o")
    # -c, NOT -fsyntax-only: instruction selection is the whole question, and
    # syntax checking stops before it. That shortcut is what let four candidates
    # be written against intrinsics the backend cannot emit.
    r = subprocess.run(
        [cc, f"-m{tgt.arch}", "-mhvx", f"-mhvx-length={tgt.hvx_bytes}B", "-mhmx",
         f"-std={CXX_STD}", "-O2", "-c", "-o", obj, path],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300)
    if r.returncode != 0:
        err = (r.stderr or r.stdout)
        if "Cannot select" in err:
            return "unselectable", "backend cannot select this intrinsic"
        m = re.search(r"error: (.+)", err)
        return "skipped", (m.group(1)[:120] if m else err[:120])

    d = subprocess.run(
        [os.path.join(os.path.dirname(cc), _exe("hexagon-llvm-objdump")), "-d", obj],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300)
    body = d.stdout if d.returncode == 0 else ""
    # Count INSTRUCTIONS, not packets. Hexagon bundles several instructions into
    # one packet, so a single-op function is often one packet -- `{ v0 = vsplat(r0);
    # jumpr r31 }` -- and a packet count of 1 called 17 perfectly good intrinsics
    # vacuous. An empty body is exactly one instruction: the `jumpr r31`.
    # One line per instruction, each starting with its address: `      0:\t...`.
    # Matching the encoding bytes as well looked tighter and was wrong -- objdump
    # separates the last byte from the mnemonic with a TAB, so a `{2} ` repetition
    # never matched the fourth byte and every probe read as empty.
    ins = [ln for ln in body.splitlines() if re.match(r"^\s*[0-9a-f]+:\s", ln)]
    if len(ins) <= 1:
        return "vacuous", "probe body optimised away -- result says nothing"
    return "selectable", f"{len(ins)} instructions"


def probe(names, sdk_root=DEFAULT_SDK_ROOT) -> dict:
    binf = find_toolchain_bin(sdk_root)
    env = toolchain_env(binf)
    cc = os.path.join(binf, _exe(COMPILER))
    tgt = _target.current()
    out = {}
    with tempfile.TemporaryDirectory(prefix="isa_probe_") as wd:
        for n in names:
            out[n] = probe_one(n, cc, env, tgt, wd)
    return out


FAMILIES = {
    # The one that mattered: everything that produces or consumes an IEEE float
    # vector, plus the qf forms that turned out to be the real datapath.
    "float": r"_V(hf|sf|qf16|qf32|bf)|_W(sf|hf|qf32)|Vhf_|Vsf_|Vqf16_|Vqf32_",
    "fp16": r"(hf|qf16)",
    "fp32": r"(sf|qf32)",
    "hmx": r"mx",
    "dma": r"dm",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=sorted(FAMILIES), default=None)
    ap.add_argument("--pattern", default=None, help="regex over intrinsic names")
    ap.add_argument("--select", default=None, help="comma-separated exact names")
    ap.add_argument("--show", choices=["all", "selectable", "unselectable"],
                    default="all")
    args = ap.parse_args(argv)

    if args.select:
        names = [s.strip() for s in args.select.split(",") if s.strip()]
    else:
        pattern = args.pattern or (FAMILIES[args.family] if args.family else None)
        names = declared(pattern)
    if not names:
        print("no intrinsics matched", file=sys.stderr)
        return 2

    result = probe(names)
    counts = {"selectable": 0, "unselectable": 0, "skipped": 0, "vacuous": 0}
    for name in names:
        status, detail = result[name]
        counts[status] += 1
        if args.show == "all" or args.show == status:
            mark = {"selectable": "OK  ", "unselectable": "NO  ",
                    "skipped": "??  ", "vacuous": "!!  "}[status]
            show_detail = status in ("skipped", "vacuous")
            print(f"{mark}{name}" + (f"   {detail}" if detail and show_detail else ""))
    tgt = _target.current()
    print(f"\n{tgt.core}: {counts['selectable']} selectable, "
          f"{counts['unselectable']} UNSELECTABLE, {counts['vacuous']} vacuous, "
          f"{counts['skipped']} not probed ({len(names)} declared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
