"""Compile, run and judge one accelerated candidate.

THE ONLY CLAIM THIS PIPELINE MAKES
----------------------------------
A kernel counts when it **compiled, executed on the simulator, and passed its own
generated golden harness**. Not when C++ was emitted, not when it looks right.
Every verdict below is the observed result of running something; nothing is
inferred from the source text.

Two questions are answered, and they are kept separate because conflating them is
how a benchmark starts scoring the wrong thing:

  **Is it correct?**  The harness compares against golden vectors produced by
  running the original PyTorch module. Exact for integer dtypes, tolerance for
  float -- HVX float arithmetic goes through the non-IEEE qfloat path and
  reductions reorder, so bit-equality would fail correct kernels.

  **Did it use the hardware?**  From the static ELF via
  `hexkernels.anticheat.anticheat`, which reads the disassembly rather than the source:
  a kernel that mentions `Q6_Vw_vadd_VwVw` in a comment has not used HVX, and one
  that calls a helper emitting real intrinsics has.

Correctness gates; mechanism use is reported. A fast kernel computing the wrong
answer is a failure, and a correct kernel that ignored the accelerator is a
result -- a common and interesting one -- not an error.

WHAT IS NOT CLAIMED HERE
------------------------
Static detection proves the instruction is PRESENT, which is weaker than
executed, which is weaker than useful. `hexkernels.anticheat.anticheat_runtime` draws the
stronger conclusions from `--packet_analyze` and the PMU counters; a kernel with
HVX in a never-taken branch passes the static check and fails that one. This
module reports the static flags and, when timing is on, the runtime evidence
alongside -- never the static flag as if it were the strong claim.
"""
import os
import re
import shutil
import subprocess
import tempfile

from hexkernels.anticheat import anticheat as _ac
from hexkernels.core import target as _target
from hexkernels.core.toolchain import (BUS_PENALTY, BUS_RATIO, COMPILER, CXX_STD,
                                    DEFAULT_SDK_ROOT, _exe, find_toolchain_bin,
                                    sim_flags_for_caps, toolchain_env)

HARNESS_INCLUDE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "env", "harness")


# THE DMA DESCRIPTOR IS ALREADY REACHABLE -- DO NOT ADD `libnative/include`.
#
# This was got wrong once, in the direction of adding an include path that breaks
# builds. The full story, because the surface facts are misleading:
#
#   * `dmstart` has ZERO hits across all 29 shipped SDK PDFs, so there is no
#     user-DMA chapter to cite. That part is true.
#   * `<toolchain>/Tools/libnative/include/udma.h` exists and is NOT on the
#     compiler's default search path (that is `Tools/target/hexagon/include`), so
#     `#include <udma.h>` fails with "file not found". Also true.
#   * From which it does NOT follow that the descriptor layout is unavailable.
#     `hexagon_types.h`, already on the default path, defines
#     `hexagon_udma_descriptor_type0_t`, `type1_t` and the `HEXAGON_UDMA_*`
#     constants verbatim. It compiles with no extra flags at all.
#
# And adding `libnative/include` is actively harmful: `udma.h` guards on
# `UDMA_HDR_H` while `hexagon_types.h` guards on `HEXAGON_TYPES_H`, so a kernel
# including both -- which any HVX kernel does -- gets a hard "redefinition of
# 'hexagon_udma_descriptor_type0_s'". Two independent kernel authors hit this and
# reported it.
#
# So the guidance for a DMA author is `#include <hexagon_types.h>`, full stop.
# Nothing is hand-derived and no include path is needed.
# Pinned by `test_the_dma_descriptor_needs_no_extra_include_path`.

# The harness prints exactly one of these.
CORRECT_TOKEN = "HVXENV_CORRECT"
INCORRECT_TOKEN = "HVXENV_INCORRECT"


def _run(cmd, env, timeout):
    """subprocess with the encoding pinned.

    `encoding=`/`errors=` are not optional. Text mode without them has twice
    killed this repo's reader thread on a stray byte, which surfaced as a
    *compile failure* for a kernel that compiled fine -- the worst kind of
    wrong verdict, because it looks like a model error.
    """
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _disasm(elf, binf, env, timeout=120):
    r = _run([os.path.join(binf, _exe("hexagon-llvm-objdump")), "-d", elf],
             env, timeout)
    return r.stdout if r.returncode == 0 else ""


def mechanisms_used(disasm_text: str) -> dict:
    """Static mechanism flags, from the disassembly of the CANDIDATE ALONE.

    Sourced from the frozen detectors in `hexkernels.anticheat.anticheat` rather than
    re-implemented, so a candidate here is judged by exactly the rules the rest
    of the benchmark uses.

    MUST be given a candidate-only disassembly. Passing the linked ELF makes
    every kernel look vectorised: the C runtime and the harness pull in library
    code containing HVX, and the detectors -- correctly -- report what they are
    shown. This was observed here, not theorised: the emitted *scalar* reference
    for relu reported `hvx_compute=True` off the linked ELF while its own object
    file contained zero HVX-register instructions at -O0 through -O3. `verify()`
    therefore disassembles a separate `-c` object, which is also how
    `core.evaluate` does its ELF-level detection.
    """
    return {
        "hvx": _ac._disasm_has_hvx(disasm_text),
        "hvx_compute": _ac._disasm_has_hvx_compute(disasm_text),
        "hmx": _ac._disasm_has_hmx(disasm_text),
        "dma": _ac._disasm_has_dma(disasm_text),
        "l2fetch": _ac._disasm_has_l2fetch(disasm_text),
        "vtcm": _ac._disasm_has_vtcm(disasm_text),
    }


_TOTAL_RE = re.compile(r"Total:\s*Insns=(\d+)\s*Pcycles=(\d+)")


def parse_failure(line: str) -> dict:
    """The `HVXENV_INCORRECT` line's `key=value` fields, as ints."""
    return {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}


#: Element counts a 128-byte HVX vector holds, by element width. Used to
#: recognise `first_bad == lanes`, which is the signature of a kernel that
#: computed exactly one vector.
_LANES = (32, 64, 128)


def diagnose(f: dict) -> list:
    """Named causes consistent with the SHAPE of the bad set. May be empty.

    `errors/n/first_bad` says a kernel is wrong. It does not say why -- and every
    numeric bug in this corpus was actually diagnosed by looking at WHICH elements
    were wrong, then reasoning back to the permutation or the rounding step that
    would produce that set. This encodes that reasoning so it happens once, in the
    verdict, instead of being re-derived by whoever reads the failure.

    HOW EACH RULE IS JUSTIFIED, kept separate on purpose -- one is verified by
    reproduction and the rest are derived from what the hardware does:

    | shape                        | hypothesis              | evidence         |
    |------------------------------|-------------------------|------------------|
    | `first_bad` == a lane count  | one vector computed,    | REPRODUCED: a    |
    |                              | rest untouched          | `floor` kernel   |
    |                              |                         | capped at k<1    |
    | uniform `bad_stride` of 2    | a widening deal read as | derived from the |
    |                              | a concatenation         | deal's semantics |
    | shared residue mod 64 or 128 | a fixed wrong offset    | derived; NOT the |
    |                              | inside each vector      | batch-13 pack    |
    |                              |                         | bug (see below)  |
    | one run reaching `n - 1`     | the tail was never      | derived from the |
    |                              | written                 | loop bound       |
    | a small scattered fraction   | a rounding difference,  | matches the      |
    |                              | not a permutation       | 17.8% magic-     |
    |                              |                         | constant case    |

    TWO THINGS THIS CANNOT DO, established by trying them:

    * **A PACK-ORDER BUG NEED NOT SHOW A RESIDUE.** Swapping the halves in
      `fp32_le_Tensor`'s predicate pack gives 250 of 512 wrong with no shared
      residue and no uniform stride -- because the packed values are comparison
      results, so scrambling their positions produces an essentially random bad
      set. The batch-13 requant kernel was 97% wrong for the same reason and was
      diagnosed by reasoning about the pack's semantics, NOT from the bad set. The
      mod-residue rule is a hypothesis about fixed-offset errors and is not
      retrofitted to that bug.
    * **A BROKEN DMA USUALLY CRASHES INSTEAD.** Dropping `fp32_log`'s final
      write-back produced "no verdict token" -- a stale descriptor issued to the
      engine, not a wrong number. There is no bad set to shape, which is why the
      static gate in `forge2.lint` covers the descriptor rules and this does not.

    HYPOTHESES, NOT VERDICTS, and the wording of each says so. A pattern says
    where to look; asserting the cause would be the same overreach as the comment
    claiming a pack inverts a deal -- which was wrong in the comment before it was
    wrong in the code.
    """
    n, errs = f.get("n", 0), f.get("errors", 0)
    fb, lb = f.get("first_bad", -1), f.get("last_bad", -1)
    if not n or errs <= 0:
        return []
    out = []
    frac = errs / n

    if fb in _LANES and fb < n and errs > 1:
        # NO "and almost everything after it is wrong" CLAUSE. The first version had
        # one and it suppressed a true positive: a `floor` kernel that computed one
        # vector reported errors=250 of the 480 elements past first_bad=32, not 479,
        # because the untouched output buffer is ZEROS and floor of half the inputs
        # is 0 -- so half the unwritten tail matched by accident. The accidental
        # matches are themselves part of the signature, which is why the message
        # points at them instead of the condition excluding the case.
        out.append(
            f"first_bad={fb} is exactly one HVX vector of {fb}-lane elements -- the "
            f"loop probably computed ONE vector and left the rest of the buffer "
            f"untouched. Check the element count against the lane count (128 bytes "
            f"is 32 fp32, 64 fp16, 128 int8 lanes). Note {errs} of the "
            f"{n - fb} elements after it are wrong rather than all of them: an "
            f"unwritten buffer is zeros, and zeros coincidentally MATCH wherever "
            f"the expected value is 0")

    if f.get("uniform_stride") and f.get("bad_stride", 0) > 1:
        s = f["bad_stride"]
        out.append(
            f"every bad element is {s} apart, uniformly -- an interleave, so a "
            f"widening/narrowing lane order: a deal delivers evens low and its "
            f"inverse is a SHUFFLE, not a concatenation")

    for k in (128, 64, 32):
        r = f.get(f"mod{k}", -1)
        if r >= 0 and errs > 1:
            out.append(
                f"every bad index is congruent to {r} mod {k} -- a fixed position "
                f"within each {k}-element group, which is what a wrong pack half "
                f"or a wrong offset inside a vector looks like")
            break

    if lb == n - 1 and f.get("run_len", 0) == errs and errs < n:
        out.append(
            f"the bad elements are one contiguous run of {errs} ending at the last "
            f"element -- the tail was never written; check the loop bound and any "
            f"final partial tile")

    if 0 < frac < 0.30 and not f.get("uniform_stride") and fb not in _LANES:
        out.append(
            f"only {frac:.1%} wrong and scattered -- consistent with a ROUNDING "
            f"difference rather than a wrong permutation: check the rounding mode "
            f"(the truncating convert), and note magic-constant rounding does not "
            f"work through qf32 on v75")

    return out


def _work_done(sim_output: str) -> dict:
    """How much the program actually executed, from the simulator's own summary.

    WHY THIS IS RECORDED. Until now the pipeline could say an intrinsic was
    PRESENT (from the disassembly) and never that it DID ANYTHING. A kernel can
    contain `Q6_Vhf_vmpy_VhfVhf` in a loop that runs once, or stage a tile through
    VTCM and then read the operands from DDR anyway, and every mechanism flag would
    still be True. Instruction count answers the question the flags cannot: a
    vectorised kernel that really replaced 64 scalar operations with one vector
    operation executes far fewer instructions than the scalar reference, and no
    amount of dead intrinsic code can fake that.

    `insns` is the robust number: it is a count of instructions retired, with no
    memory model and no calibration in it. `pcycles` WITHOUT `--timing` is the
    simulator's idealised issue count, NOT a memory-modeled cost -- it is recorded
    because it is free, and it must not be reported as a cycle cost.

    Returns {} when the summary is absent (a crash before exit), so a caller can
    tell "not measured" from "measured zero".
    """
    m = _TOTAL_RE.search(sim_output)
    if not m:
        return {}
    return {"insns": int(m.group(1)), "pcycles": int(m.group(2))}


def _param_names(signature: str) -> list:
    """The parameter NAMES of a `candidate_kernel` declaration, in order.

    `void candidate_kernel(const float *v_x, float *out0)` -> ["v_x", "out0"].
    Used to write a call to it, so it has to agree with the declaration exactly;
    it reads the same `kernel_signature(graph)` string the reference and the
    candidate were both compiled against, so there is one source of truth.
    """
    inner = signature[signature.index("(") + 1:signature.rindex(")")]
    names = []
    for part in inner.split(","):
        tok = part.strip().replace("*", " ").split()
        if tok:
            names.append(tok[-1])
    return names


def double_call_probe(kernel_source: str, signature: str) -> str:
    """The same kernel, called TWICE, so the difference is one execution of it.

    WHY NOT AN EMPTY KERNEL. The obvious way to measure what the harness costs is
    to link it against a `candidate_kernel` with an empty body. That was tried and
    it is WRONG, measurably: an empty kernel leaves every output at zero, so the
    harness takes its MISMATCH path on every element instead of its passing path,
    and the two paths do not cost the same. Measured on batch 9, the empty-kernel
    "overhead" came out LARGER than the entire candidate run it was supposed to be
    a component of -- 31,517,473 against 31,023,868 for fp16_bmm -- which cannot
    happen for a real overhead and is the signature of having measured a different
    program.

    This probe runs the identical program with one extra call to the identical
    function. Both runs PASS (the kernel is a pure function of its inputs, so
    calling it twice writes the same outputs), so the harness follows the same
    path in both, and

        work(kernel)   = insns(doubled) - insns(single)      exact
        harness(pass)  = insns(single)  - work(kernel)
        work(reference)= insns(reference) - harness(pass)

    with no modelling anywhere. Instruction count is cache-independent, so the
    second call costs exactly the same as the first even though it runs warm --
    which is why this works for `insns` and would NOT work for cycles.

    The probe passing is itself the precondition being checked: a kernel that
    accumulates into its output rather than writing it would produce a DIFFERENT
    answer on the second call, the harness would report INCORRECT, and the
    measurement is discarded rather than reported.
    """
    nl = chr(10)
    args = ", ".join(_param_names(signature))
    renamed = kernel_source.replace("candidate_kernel", "candidate_kernel_once")
    wrapper = (f"{nl}{nl}/* double-call work probe: see verify.double_call_probe */"
               f'{nl}extern "C" {signature} {{{nl}'
               f"    candidate_kernel_once({args});{nl}"
               f"    candidate_kernel_once({args});{nl}}}{nl}")
    return renamed + wrapper


def work_ratio(reference: dict, candidate: dict, probe: dict = None) -> dict:
    """Reference-over-candidate instruction counts, with the harness subtracted.

    WHY THE RAW RATIO IS WORTHLESS HERE, measured rather than suspected. On
    batch 8 the five raw ratios were 1.02, 1.02, 1.03, 1.02, 1.03 -- for kernels
    whose inner loops had just been replaced by 32-lane vector code. Both runs
    execute the SAME harness, and that harness decodes a base64 image of every
    golden buffer and then compares it element by element; at these sizes it is
    tens of millions of instructions against a kernel of a few hundred thousand,
    so it drags every ratio to 1. Reporting that as "the speedup" would understate
    every kernel in this corpus by two orders of magnitude.

    `probe` is a run of the identical program with the candidate called twice (see
    `double_call_probe`), which gives the candidate's own instruction count as a
    difference and the harness's cost by subtraction. Both quantities are exact.

    Without a probe, `ratio` is still reported and is still a LOWER BOUND, and
    `delta` (reference minus candidate) is the part attributable to the kernels,
    since the shared overhead cancels in the difference.
    """
    ri, ci = reference.get("insns"), candidate.get("insns")
    if not ri or not ci:
        return {}
    out = {"ref_insns": ri, "cand_insns": ci,
           "ratio": round(ri / ci, 2), "delta": ri - ci}
    pi = (probe or {}).get("insns")
    if not pi:
        return out
    if not (probe or {}).get("correct"):
        out["kernel_ratio_note"] = ("the double-call probe did not pass: the "
                                    "kernel is not idempotent, so its work "
                                    "cannot be measured this way")
        return out
    cand_work = pi - ci
    harness = ci - cand_work
    ref_work = ri - harness
    if cand_work <= 0 or ref_work <= 0:
        out["kernel_ratio_note"] = (f"probe delta {cand_work} is not a usable "
                                    "kernel cost; harness not subtracted")
        return out
    out["harness_insns"] = harness
    out["cand_work"] = cand_work
    out["ref_work"] = ref_work
    out["kernel_ratio"] = round(ref_work / cand_work, 2)
    return out


def compile_flags(cc, tgt, rd_build=False) -> list:
    """The compile line, in ONE place because two callers must never disagree.

    `verify` uses it twice already -- once to link the candidate with its harness
    and once for the separate `-c` compile that mechanism detection disassembles --
    and its own comment says "same flags as the link so the two can never disagree
    about what was judged". `mechanisms_of` is now a third caller, so the line is
    factored out rather than copied a third time.
    """
    base = [cc, f"-m{tgt.arch}", "-mhvx", f"-mhvx-length={tgt.hvx_bytes}B",
            "-mhmx", f"-std={CXX_STD}", "-O2"]
    if rd_build:
        base += ["-I", HARNESS_INCLUDE]
    else:
        base += ["-DFORGE2_BUILD=1"]
    return base


def mechanisms_of(kernel_source: str, *, name="candidate",
                  sdk_root=DEFAULT_SDK_ROOT, timeout=300,
                  rd_build=False) -> dict:
    """Which mechanisms a kernel USES, without running it. No simulator.

    WHY THIS CAN EXIST AT ALL, and it is the anti-cheat's central property: a
    mechanism claim is decided by DISASSEMBLING the candidate's own object, not by
    observing execution. `verify` already does exactly this, before the simulator
    starts, and the result does not depend on anything the simulator produces.

    WHAT IT BUYS. `genuine` is `correct AND an entitled mechanism fires`. The
    correctness half costs a simulator run -- ~12 minutes for a T2 kernel at ~2.2
    billion instructions, and simulation is serial here. This half costs a compile
    and an objdump: ~0.4 seconds. So the mechanism question can be answered across
    a whole eval set in minutes while correctness is still grinding, and a sweep
    that finds NO mechanism firing settles `genuine = 0` for those attempts on its
    own, because a conjunction with a false half is false whatever the other half.

    WHAT IT IS NOT. It is not a correctness verdict and must never be recorded as
    one. A kernel that computes nonsense with beautiful HVX code passes this and
    fails the benchmark.

    One asymmetry to carry with any number derived from this: `vtcm` here is the
    STATIC detector (`anticheat._disasm_has_vtcm`), not the runtime PMU counter.
    That difference is a disclosure item in PLAN.md section 5.F.
    """
    tgt = _target.current()
    binf = find_toolchain_bin(sdk_root)
    env = toolchain_env(binf)
    cc = os.path.join(binf, _exe(COMPILER))
    out = {"scanned": False, "compiled": False, "mechanisms": {},
           "error_text": ""}
    tmp = tempfile.mkdtemp(prefix="forge2_mech_")
    try:
        kpath = os.path.join(tmp, "kernel.cpp")
        with open(kpath, "w", encoding="utf-8") as f:
            f.write(kernel_source)
        obj = os.path.join(tmp, f"{name}.o")
        try:
            r = _run(compile_flags(cc, tgt, rd_build) + ["-c", "-o", obj, kpath],
                     env, timeout)
        except subprocess.TimeoutExpired:
            out["error_text"] = "mechanism compile timed out"
            return out
        if r.returncode != 0:
            # Fail CLOSED and say so, exactly as `verify` does: no object, no
            # mechanism claims. This is distinguishable from "compiled and used
            # nothing", and conflating them would credit a kernel that never built
            # with having honestly declined the accelerator.
            out["error_text"] = (r.stderr or r.stdout or "")[:4000]
            return out
        out["compiled"] = True
        out["mechanisms"] = mechanisms_used(_disasm(obj, binf, env))
        out["scanned"] = True
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify(kernel_source: str, harness_source: str, *, name="candidate",
           timing=False, sdk_root=DEFAULT_SDK_ROOT, timeout=900,
           workdir=None, rd_build=False, bus_penalty=None, bus_ratio=None) -> dict:
    """Compile a candidate against its harness, run it, and report.

    `kernel_source` is the candidate translation unit (it must define
    `extern "C" void candidate_kernel(...)`); `harness_source` is the generated
    `main()` that checks it against golden vectors.

    Returns a verdict dict whose shape is stable whatever happens, so a caller
    never has to distinguish "field absent" from "step not reached".
    """
    tgt = _target.current()
    verdict = {
        "name": name, "compiled": False, "ran": False, "correct": False,
        "mechanisms": {}, "error_text": "", "stdout": "",
        "target": tgt.core, "timing": timing, "timed_out": False,
    }

    binf = find_toolchain_bin(sdk_root)
    env = toolchain_env(binf)

    tmp = workdir or tempfile.mkdtemp(prefix="forge2_verify_")
    os.makedirs(tmp, exist_ok=True)
    kpath = os.path.join(tmp, "kernel.cpp")
    hpath = os.path.join(tmp, "harness.cpp")
    elf = os.path.join(tmp, f"{name}.elf")
    with open(kpath, "w", encoding="utf-8") as f:
        f.write(kernel_source)
    with open(hpath, "w", encoding="utf-8") as f:
        f.write(harness_source)

    cc = os.path.join(binf, _exe(COMPILER))
    # One flag list for both compiles. Kept in a single variable so the linked
    # binary and the object the mechanisms are read from can never drift -- if
    # they differed in -O level they would disagree about what was judged.
    #
    # NOTE WHAT IS ABSENT: no `-I hexbench/env/harness`. Every header in that
    # directory (`hmx_helpers.h`, `harness_common.h`, ...) is R&D material, and a
    # forge v2 translation unit that reaches into it has a weaker provenance chain
    # than this pipeline claims. The first attempt at closing that made the
    # headers *forbidden* -- the path stayed, plus `-DFORGE2_BUILD=1` for
    # `hmx_helpers.h` to `#error` on. Removing the path instead makes them
    # UNREACHABLE: `#include "harness_common.h"` is now a file-not-found, which no
    # amount of care or carelessness can turn into a successful build. A kernel
    # derives from the reference, the schedule and the vendor intrinsic headers
    # (`<hexagon_types.h>`, `<hexagon_protos.h>`, `<hvx_hexagon_protos.h>`,
    # `<hmx_hexagon_protos.h>`), which live on the toolchain's own default path.
    # The generated harness is self-contained for the same reason -- it emits its
    # own tolerance compares and HMX enable (`forge.frontend.oracle.harness_c`).
    #
    # `-DFORGE2_BUILD=1` is kept as the second layer, for a candidate that reaches
    # the header by some other route (a copied path, a stale local include dir).
    # `provenance.rd_leaks` is the third: it catches a copied BODY, which neither
    # of the other two can.
    #
    # `rd_build=True` restores the include path and drops the define, for exactly
    # one purpose: letting the R&D side's own tests exercise `hmx_helpers.h`,
    # which they must be able to do or the helper goes back to being shipped
    # untested -- which is how it came to return all zeros. It is NOT an escape
    # hatch for candidates: nothing on the candidate path
    # (`run_batch.verify_candidates`, `check`) passes it, and `test_provenance.py`
    # asserts that.
    base = compile_flags(cc, tgt, rd_build)
    try:
        r = _run(base + ["-o", elf, hpath, kpath], env, timeout)
    except subprocess.TimeoutExpired:
        verdict["error_text"] = "compile timed out"
        verdict["timed_out"] = True
        return verdict
    if r.returncode != 0:
        verdict["error_text"] = (r.stderr or r.stdout)[:8000]
        return verdict
    verdict["compiled"] = True

    # Mechanism detection uses a SEPARATE `-c` compile of the candidate alone.
    # Disassembling the linked ELF would attribute the C runtime's and the
    # harness's library code to the candidate -- measured: the scalar relu
    # reference reports hvx_compute off the link and has no HVX whatsoever in
    # its own object. Same flags as the link so the two can never disagree about
    # what was judged.
    obj = os.path.join(tmp, f"{name}.o")
    try:
        ro = _run(base + ["-c", "-o", obj, kpath], env, timeout)
    except subprocess.TimeoutExpired:
        ro = None
    if ro is not None and ro.returncode == 0:
        verdict["mechanisms"] = mechanisms_used(_disasm(obj, binf, env))
    else:
        # Fail closed: no object, no mechanism claims.
        verdict["mechanisms"] = {k: False for k in
                                 ("hvx", "hvx_compute", "hmx", "dma",
                                  "l2fetch", "vtcm")}
        verdict["error_text"] = "mechanism detection compile failed"

    # HMX must be enabled on BOTH sides. Compiling with `-mhmx` only lets the
    # instructions be encoded; the simulator additionally needs `--mhmx 2` or it
    # will not execute them. Omitting the sim half made HMX unreachable through
    # this harness no matter what a candidate wrote -- so an HMX kernel would
    # have failed at runtime and been recorded as the author's mistake. Caught
    # by an author who investigated why the mechanism was unreachable and read
    # the verifier rather than assuming its harness was correct.
    #
    # Sourced from `toolchain.sim_flags_for_caps` rather than spelled out here,
    # so the compile flags and the sim flags cannot drift apart again.
    sim = [os.path.join(binf, _exe("hexagon-sim")), f"--m{tgt.arch}"]
    sim += sim_flags_for_caps(["hmx"])
    if timing:
        # The two knobs the simulator UG names as governing timing accuracy
        # ("bus clock ratio and bus delay will impact the overall accuracy").
        # `toolchain.BUS_PENALTY`/`BUS_RATIO` pin REPRESENTATIVE defaults, not
        # device-matched values, which is why they are overridable: any claim that
        # depends on them -- whether a prefetch or a DMA staging pays for itself --
        # is a claim about one bus configuration, and the honest way to state it is
        # to show how it moves when the bus moves. Defaults preserve the pinned
        # values exactly, so every existing caller is unaffected.
        bp = BUS_PENALTY if bus_penalty is None else bus_penalty
        br = BUS_RATIO if bus_ratio is None else bus_ratio
        sim += ["--timing", "--buspenalty", str(bp), "--busratio", str(br)]
        verdict["bus_penalty"], verdict["bus_ratio"] = bp, br
    sim.append(elf)
    try:
        r = _run(sim, env, timeout)
    except subprocess.TimeoutExpired:
        # A TIMEOUT IS NOT A WRONG ANSWER. Flagged rather than left to the caller
        # to recognise from `error_text`, because a reworded message must not be
        # able to turn an exhausted budget back into a failed measurement. See
        # `rung0.grade_one`, which refuses to record a verdict when this is set.
        verdict["error_text"] = f"simulator timed out after {timeout}s"
        verdict["timed_out"] = True
        return verdict
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    verdict["ran"] = True
    verdict["stdout"] = out[-8000:]
    verdict.update(_work_done(out))

    if CORRECT_TOKEN in out:
        verdict["correct"] = True
    elif INCORRECT_TOKEN in out:
        line = next((l for l in out.splitlines() if INCORRECT_TOKEN in l), "")
        verdict["error_text"] = line.strip()
        verdict["failure"] = parse_failure(line)
        verdict["diagnosis"] = diagnose(verdict["failure"])
        if verdict["diagnosis"]:
            verdict["error_text"] += "\n  likely cause: " + "; ".join(
                verdict["diagnosis"])
    else:
        # Fail closed. No verdict token means the harness did not reach its
        # comparison -- a crash, a hang, or a kernel that never returned. That
        # is not a pass.
        verdict["error_text"] = "no verdict token in simulator output"
    return verdict
