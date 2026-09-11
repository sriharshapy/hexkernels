#!/usr/bin/env python3
"""Behavior-frozen reward core: compile -> simulate -> capture for one candidate.

Takes a candidate C kernel (implementing candidate_kernel from the task's
kernel_api.h), wraps it in the task's fixed harness, compiles with hexagon-clang,
runs on hexagon-sim, and reports the full ``feedback`` dict (compiled / correct /
errors / cycles / anti-cheat verdicts / VLIW + roofline + PMU diagnostics).

This is the environment's step() in miniature and the credibility core of the
whole benchmark: the returned ``feedback`` dict (its keys, defaults, all sim/PMU
parsing, the compile+objdump+sim flow, and the ``__main__`` CLI) is LOGIC-FROZEN
and byte-identical to the old ``m1_driver/evaluate.py``. This rebuild only:
  * imports the toolchain plumbing from ``core.toolchain``,
  * imports the ELF/source anti-cheat from ``hexkernels.anticheat.anticheat``,
  * imports the roofline model as ``core.roofline``,
  * repoints the task/common directories at the ``hexbench`` package layout.
No number, string, regex, or branch was changed.

DEFAULT_TASK is ``hvx_vadd_i8`` (the v6 analog of the old ``int8_vadd`` task,
which does not exist in v6). It only affects the bare CLI: every v6-path caller
passes an explicit ``task_dir=``.

Usage:
    python -m core.evaluate kernels/candidate.c
    python -m core.evaluate --selftest
    python -m core.evaluate kernels/foo.c --json
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
from typing import Optional

from hexkernels.core.toolchain import (
    DEFAULT_SDK_ROOT,
    DSP_ARCH,
    TIMING_MODE,
    BUS_PENALTY,
    BUS_RATIO,
    HVX_CFLAGS,
    SIM_TIMEOUT_S,
    SIM_TIMEOUT_MAX_S,
    run,
    toolchain_env,
    find_toolchain_bin,
    _exe,
    cflags_for_caps,
    sim_flags_for_caps,
    COMPILER,
)
from hexkernels.anticheat import anticheat
from hexkernels.core import target as _target
from hexkernels.anticheat import anticheat_runtime as _acr
from hexkernels.core import roofline as _roofline

HERE = os.path.dirname(os.path.abspath(__file__))
# Task dirs live at <repo_root>/data/v6/tasks (created in a later task; this
# path is not validated at import time). <repo_root> = the dir three levels up
# from this file: .../HVX-clean/hexbench/env/evaluate.py -> .../HVX-clean.
TASKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "v6", "tasks",
)
# Shared -I include dir: the two headers (harness_common.h, hmx_helpers.h) that
# every task's harness.c includes. They live next to this module.
COMMON_DIR = os.path.join(HERE, "harness")
DEFAULT_TASK = "hvx_vadd_i8"

_TIMING_MODEL = None
_WARNED: set = set()


def _warn_once(msg: str) -> None:
    """Print a diagnostic once per process. A per-task warning would print on
    every one of hundreds of evaluations and be tuned out; once is visible."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[hexbench] WARNING: {msg}", file=sys.stderr)


class StaleTimingModel(ValueError):
    """``timing_model.json`` describes a different part than the active target.

    Raised rather than tolerated. The roofline's bandwidth ceilings are properties
    of a specific memory system, so applying one part's model to another does not
    degrade gracefully -- it yields an ``eta`` that is confidently wrong with no
    outward sign. v75 has twice the VTCM of v68 at a different base; nothing in
    the arithmetic notices.
    """


def timing_model() -> dict:
    """The calibrated memory model, or raise if it is not this target's.

    Regenerate for a new target with::

        python -m core.calibrate --out hexbench/env/timing_model.json
    """
    global _TIMING_MODEL
    if _TIMING_MODEL is None:
        with open(os.path.join(HERE, "timing_model.json")) as f:
            _TIMING_MODEL = json.load(f)
    core = _target.current().core
    if _TIMING_MODEL.get("arch") != core:
        raise StaleTimingModel(
            f"timing_model.json was calibrated for {_TIMING_MODEL.get('arch')!r} "
            f"but the active target is {core!r}; regenerate it with "
            "`python -m core.calibrate` or set HEXBENCH_TARGET to match")
    return _TIMING_MODEL


def _engine_for(fb: dict) -> str:
    return "hmx" if fb.get("used_hmx") else "hvx"


def attach_roofline(fb: dict, task_dir: str) -> None:
    """Populate fb with roofline fields (eta etc.) from the task's work facts.
    Adds keys even when not computable (set to None) so the dict shape is stable."""
    for k in ("achieved", "AI", "binding_level", "ceiling", "eta", "eta_raw",
              "eff_bw", "bw_asymptotic"):
        fb.setdefault(k, None)
    fb.setdefault("roofline_unavailable", None)
    try:
        with open(os.path.join(task_dir, "spec.json"), encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError):
        fb["roofline_unavailable"] = "no readable spec.json"
        return
    uo, wb = spec.get("useful_ops"), spec.get("work_bytes")
    if not (uo and wb):
        fb["roofline_unavailable"] = "spec lacks useful_ops/work_bytes"
        return
    try:
        model = timing_model()
    except StaleTimingModel as exc:
        # Reported, not swallowed. A stale model does not make the roofline
        # slightly off -- it makes it wrong with no outward sign -- so the
        # absence of the fields is recorded with its reason rather than looking
        # like an ordinary "spec had no work facts" miss.
        fb["roofline_unavailable"] = str(exc)
        _warn_once(str(exc))
        return
    except (OSError, ValueError, KeyError) as exc:
        fb["roofline_unavailable"] = f"timing model unreadable: {exc}"
        return
    try:
        r = _roofline.roofline_efficiency(
            useful_ops=uo, work_bytes=wb, kernel_cycles=fb.get("kernel_cycles"),
            dtype=spec.get("dtype", "int8"), engine=_engine_for(fb), model=model)
    except (OSError, ValueError, KeyError) as exc:
        fb["roofline_unavailable"] = f"roofline failed: {exc}"
        return
    fb.update(r)


def _sim_timeout_for(task_dir: str) -> int:
    """Size-aware sim timeout (seconds). Small tasks keep the 60s base; large /
    bandwidth-bound XL tasks (big work_bytes) get proportionally more so a
    *correct* prefetched kernel isn't scored as a spurious timeout — the flat
    60s cut off n=500K XL kernels mid-run. Clamped to [60, 900] so infinite
    loops still die. work_bytes is the size signal (falls back to n, then base).

    Calibration: small int8_vadd work_bytes=3K -> 60s; int8_vadd_xl work_bytes=1.5M
    -> ~444s. Divisor 4096 (~1s per 4 KB of moved data); the ceiling caps it.

    DIVISOR MARGIN MATTERS, not just the ceiling. At the old 8192, dma_ping_pong_conv1d_i8
    (params.n=1.2M) got a 206s budget against a measured 204.5s serial runtime -- 1.5s of
    headroom, so it passed alone and timed out under any parallel load. A budget within a
    few percent of true runtime is flaky by construction, and flakiness here is
    indistinguishable from a wrong kernel. 4096 gives that task ~1.7x headroom and the
    heaviest measured DMA experts 2x+.

    UNKNOWN SIZE => ASSUME LARGE. Some v6-origin specs carry their dimensions only
    inside ``params`` (e.g. ``params.n``, or R/W dims) and have no top-level
    ``work_bytes``/``n`` at all -- 9 of the 64 holdout tasks. Treating "no size
    signal" as "small" gave them the flat 60s base and scored four *correct* DMA
    experts (i8_gather_vtcm, i8_layernorm_dma, i8_pooling_dma_doublebuf,
    i8_streaming_conv_dma; measured 66-181s under timing) as spurious timeouts.
    Absent metadata we cannot prove a task is small, and the asymmetry is not
    symmetric: calling a correct kernel wrong corrupts the benchmark, while an
    over-generous budget only delays killing a hang -- which the ceiling still
    bounds. So fall back to ``params.n`` when present, else the ceiling."""
    try:
        with open(os.path.join(task_dir, "spec.json"), encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError):
        return SIM_TIMEOUT_S
    size = spec.get("work_bytes") or spec.get("n")
    if not size:
        params = spec.get("params") or {}
        size = params.get("n") if isinstance(params, dict) else None
        if not size:
            return SIM_TIMEOUT_MAX_S       # no size signal anywhere -> assume large
    scaled = SIM_TIMEOUT_S + size / 4096.0
    return int(max(SIM_TIMEOUT_S, min(scaled, SIM_TIMEOUT_MAX_S)))


# --- Per-task capability flags (v4) --------------------------------------
# A task's spec.json may declare an optional "caps" list, e.g. ["hmx"].
# Absent/empty => current HVX-only behavior (every v3 task is unchanged).
def task_caps_in(task_dir: str) -> list:
    """Like task_caps but for an explicit task directory (v4 generated tasks)."""
    try:
        with open(os.path.join(task_dir, "spec.json"), encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError):
        return []
    caps = spec.get("caps", [])
    return list(caps) if isinstance(caps, (list, tuple)) else []


def task_caps(task_id: str) -> list:
    """Return the spec.json "caps" list for a task ([] if missing/unreadable)."""
    return task_caps_in(os.path.join(TASKS_DIR, task_id))


# --- ISA-native (VLIW) metrics -------------------------------------------
# These are the paper's differentiator vs latency-only prior work
# (AscendKernelGen / NPUKernelBench): we report VLIW-level behaviour, not just
# cycles/speedup. Both are computed from data we already collect (sim summary +
# the candidate objdump done for the used_hvx anti-cheat) -- no extra sim runs.
#
# Dynamic (whole-program runtime): the hexagon-sim end-of-run summary prints a
# per-HW-thread line "T0: Insns=N Packets=M" (and "Total: ... Pcycles=X"). We
# sum Insns/Packets across the active threads.
_SIM_THREAD = re.compile(r"\bT\d+:\s*Insns=(\d+)\s+Packets=(\d+)")


def _parse_sim_counters(combined: str):
    """Sum per-thread (Insns, Packets) from the sim summary; (None, None) if absent."""
    insns = packets = 0
    found = False
    for m in _SIM_THREAD.finditer(combined):
        insns += int(m.group(1))
        packets += int(m.group(2))
        found = True
    return (insns, packets) if found else (None, None)


_KCYC = re.compile(r"HVXENV_KCYCLES kernel=(\d+)")


def parse_kernel_cycles(combined: str) -> Optional[int]:
    """Kernel-only Pcycles from the harness pcycle delta; None if absent."""
    m = _KCYC.search(combined)
    return int(m.group(1)) if m else None


# --- PMU memory-system counters (REPORTED-ONLY diagnostics) --------------
# `hexagon-sim --pmu_statsfile <file>` (timing mode only) writes a small
# plain-text dump of ~230 named PMU events as `0xHEX:NAME:value` lines. These
# are surfaced into the feedback dict purely as memory-system evidence for
# the paper's mechanism story (cache misses / DDR traffic / l2fetch
# effectiveness) -- they are whole-program (same dilution caveat as `cycles`
# vs `kernel_cycles`) and must NEVER be used by reward.py. See
# .superpowers/sdd/mem-counter-investigation.md for the counter-name source.
_PMU_LINE = re.compile(r"^0x[0-9a-fA-F]+:([A-Za-z0-9_]+):(\d+)$", re.MULTILINE)


def parse_pmu_stats(text: str) -> dict:
    """Parse a --pmu_statsfile dump into {EVENT_NAME: int}. Pure (no I/O).

    Ignores blank/malformed lines instead of raising, so a partially-garbled
    dump still yields whatever counters did parse cleanly (fail-open)."""
    if not text:
        return {}
    return {name: int(val) for name, val in _PMU_LINE.findall(text)}


def _read_pmu_stats(path: str) -> Optional[dict]:
    """Read+parse a --pmu_statsfile dump from disk. None if missing/unreadable
    (fail-open: a missing/garbled PMU file must never fail the candidate)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    return parse_pmu_stats(text)


def _rate(miss: int, total: int) -> float:
    """misses/accesses, 0.0 when the denominator is 0 (per spec)."""
    return round(miss / total, 6) if total else 0.0


def _derive_mem_fields(pmu: Optional[dict]) -> dict:
    """Derive the ~6 named memory-system diagnostic fields from raw PMU counters.

    REPORTED-ONLY: these are diagnostics for the paper's memory-system story;
    they must never feed reward.py (spec Sec. 2). Whole-program dilution
    applies (see `pmu_wholeprogram`), same caveat as `cycles` vs
    `kernel_cycles` -- there is no kernel-isolated PMU split.

    `pmu` falsy (None -- no file/functional mode/timing off; or {} -- file
    present but nothing parsed) means "no data at all": every field is None,
    never a misleading 0.0. When real counters ARE present but a specific
    rate's denominator is 0 (e.g. a kernel that issued no l2fetch), that
    individual field is 0.0/None per-counter as documented inline below --
    this is "we have data and it says zero", a different fact from "no data".

    Field-name caveats (paper-defensibility; see
    .superpowers/sdd/mem-counter-verification.md):
    - `scalar_l1d_miss_rate`: scalar-only (COMMITTED_LOADS/STORES exclude HVX
      coprocessor accesses).
    - `ddr_read_requests`/`ddr_write_requests`: AXI primary-master read/write
      requests (AXI_READ/WRITE_REQUEST); a DDR proxy -- the docs don't
      literally say DDR.
    - `vtcm_accesses`: UDMA_VTCM_RD/WR -- real+deterministic but UNDOCUMENTED
      in the SDK PRMs; inferred as VTCM-DMA transactions. Cite as uncertain.
    """
    fields = {
        "scalar_l1d_miss_rate": None, "l2_miss_rate": None,
        "ddr_read_requests": None, "ddr_write_requests": None,
        "l2fetch_hit_rate": None, "l2fetch_commands": None,
        "vtcm_accesses": None, "hvx_l2_miss": None,
        "pmu_wholeprogram": True,
    }
    if not pmu:
        return fields

    loads = pmu.get("COMMITTED_LOADS", 0)
    stores = pmu.get("COMMITTED_STORES", 0)
    l1d_miss = pmu.get("DCACHE_DEMAND_MISS", 0) + pmu.get("DCACHE_STORE_MISS", 0)
    # scalar-only: COMMITTED_LOADS/STORES exclude HVX coprocessor loads/stores
    # (HVX data movement shows up separately in hvx_l2_miss).
    fields["scalar_l1d_miss_rate"] = _rate(l1d_miss, loads + stores)

    l2_acc = pmu.get("L2_DU_READ_ACCESS", 0) + pmu.get("L2_DU_STORE_ACCESS", 0)
    l2_miss = pmu.get("L2_DU_READ_MISS", 0) + pmu.get("L2_DU_STORE_MISS", 0)
    fields["l2_miss_rate"] = _rate(l2_miss, l2_acc)

    # DDR proxy: AXI primary-master read/write requests -- the SDK docs don't
    # literally say "DDR", this is inferred from the AXI bus being the path
    # to DDR on this target.
    fields["ddr_read_requests"] = pmu.get("AXI_READ_REQUEST")
    fields["ddr_write_requests"] = pmu.get("AXI_WRITE_REQUEST")

    # l2fetch_hit_rate is None (not 0.0) when L2FETCH_ACCESS==0: the kernel
    # issued no l2fetch at all, which is a different fact from "issued one
    # and it missed every time".
    l2f_acc = pmu.get("L2FETCH_ACCESS", 0)
    l2f_miss = pmu.get("L2FETCH_MISS", 0)
    fields["l2fetch_hit_rate"] = round(1 - l2f_miss / l2f_acc, 6) if l2f_acc else None
    fields["l2fetch_commands"] = pmu.get("L2FETCH_COMMAND")

    # UNDOCUMENTED in the SDK PRMs: UDMA_VTCM_RD/WR are real+deterministic
    # counters but their exact semantics aren't specified there; inferred as
    # VTCM-DMA transaction counts. Cite as uncertain.
    fields["vtcm_accesses"] = pmu.get("UDMA_VTCM_RD", 0) + pmu.get("UDMA_VTCM_WR", 0)
    fields["hvx_l2_miss"] = pmu.get("HVXLD_L2_MISS", 0) + pmu.get("HVXST_L2_MISS", 0)
    return fields


def _parse_sim_output(combined: str, fb: dict, timing: bool) -> dict:
    """Parse hexagon-sim output into fb in-place. Returns fb for chaining.

    Shared by the single-sim and two-sim (fast_incorrect) code paths so the
    parsing logic stays in one place and cannot drift between them.
    """
    m = re.search(r"Pcycles=(\d+)", combined)
    if m:
        fb["cycles"] = int(m.group(1))

    fb["kernel_cycles"] = parse_kernel_cycles(combined)

    d_insns, d_pkts = _parse_sim_counters(combined)
    if d_insns is not None:
        fb["insns"], fb["packets"] = d_insns, d_pkts
        if d_pkts > 0:
            fb["pack_p_dyn"] = round(d_insns / d_pkts, 4)
            if fb["cycles"]:
                fb["cyc_per_packet"] = round(fb["cycles"] / d_pkts, 4)
                fb["stall_rate"] = round((fb["cycles"] - d_pkts) / fb["cycles"], 4)

    if "HVXENV_CORRECT" in combined:
        fb["correct"] = True
        fb["errors"] = 0
        mn = re.search(r"HVXENV_CORRECT errors=\d+ n=(\d+)", combined)
        if mn:
            fb["n"] = int(mn.group(1))
    else:
        m = re.search(r"HVXENV_INCORRECT errors=(\d+)(?: n=(\d+))?", combined)
        if m:
            fb["errors"] = int(m.group(1))
            if m.group(2):
                fb["n"] = int(m.group(2))
            fb["error_text"] = next(
                (ln.strip() for ln in combined.splitlines() if "HVXENV_INCORRECT" in ln),
                "",
            )
        else:
            fb["error_text"] = (
                "simulator produced no HVXENV result marker (crash or no output)\n"
                + combined.strip()[:500]
            )
    return fb


def _new_feedback(task_id, candidate_c):
    """Construct the feedback dict with every key at its pre-run default.

    Extracted from evaluate() so the frozen dict shape (and the three new
    mechanism flags) can be unit-tested without the SDK toolchain. Callers in
    evaluate() overwrite sim_mode/target immediately based on the `timing`
    argument -- their defaults here are placeholders only.
    """
    return {
        "task_id": task_id,
        "candidate": os.path.abspath(candidate_c),
        "compiled": False, "correct": False,
        "used_hvx": False,  # ELF verdict, set after the candidate-object compile
        "used_hvx_compute": False,  # ELF verdict: a genuine HVX vector *compute* op
                                    # (not load-only) — the reward accel gate (design §4)
        "used_hmx": False,  # ELF verdict, set after the candidate-object compile
        "used_dma": False,        # STATIC ELF flag, set from the same candidate-only
                                   # objdump disasm as used_hvx/used_hmx (dm* insns)
        "used_l2fetch": False,    # STATIC ELF flag, set from the same disasm (l2fetch insn)
        "used_vtcm": False,       # RUNTIME PMU flag (vtcm_accesses > 0), set AFTER the
                                   # timing-mode sim/PMU parse -- NOT an ELF verdict
        "used_hvx_src": anticheat.detect_used_hvx(candidate_c),  # source heuristic (audit)
        # --- RUNTIME mechanism verdicts (see _set_genuine_flags) --------------
        # The used_* flags above answer "is the instruction PRESENT". These answer
        # "did it EXECUTE, and is it not provably futile" -- a strictly stronger
        # claim, from --packet_analyze per-packet commits plus PMU work counters.
        # FAIL-CLOSED: all False unless genuine_evidence is True, which requires a
        # timing-mode run. All-False with genuine_evidence False means "not
        # demonstrated", NOT "not present" -- read used_* for presence.
        # Does NOT detect a mechanism that executes and whose result is then
        # discarded; conformance figures remain upper bounds.
        "genuine_evidence": False,   # was a packet profile available and parsed?
        "genuine_hvx_compute": False,
        "genuine_hmx": False,
        "genuine_dma": False,
        "genuine_l2fetch": False,
        "genuine_vtcm": False,
        "packets_profiled": None,    # packets in the profile (whole program)
        "packets_executed": None,    # of those, packets with commits > 0
        "l2fetch_futile": None,      # L2FETCH_COMMAND_KILLED/OVERWRITE/CREDIT_FAIL/
                                     # PAGE_TERMINATION/DROP -- issued, achieved nothing
        "errors": None, "n": None, "cycles": None, "kernel_cycles": None, "error_text": "", "timed_out": False,
        "sim_mode": "timing",
        "target": _target.current().core,
        # ISA-native VLIW metrics (paper differentiator: VLIW-level behaviour,
        # not just latency/speedup). Dynamic = whole-program runtime (sim summary);
        # static = candidate-only disasm (clean, excludes harness/CRT).
        "insns": None, "packets": None,            # dynamic raw counts
        "pack_p_dyn": None,                        # Insns/Packets in [1,4]; VLIW packing density
        "cyc_per_packet": None,                    # Pcycles/Packets
        "stall_rate": None,                        # (Pcycles-Packets)/Pcycles
        "static_insns": None, "static_packets": None, "static_hvx_insns": None,
        "pack_p_static": None,                     # candidate insns/packets
        "vec_frac": None,                          # HVX vector instrs / candidate insns
        # Roofline efficiency (populated by attach_roofline after sim).
        "achieved": None, "AI": None, "binding_level": None,
        "ceiling": None, "eta": None, "eta_raw": None,
        "eff_bw": None, "bw_asymptotic": None,
        # Why the roofline fields above are None, when they are. Distinguishes an
        # ordinary miss (spec carried no work facts) from a STALE MODEL -- a
        # timing_model.json calibrated for a different part, which would
        # otherwise produce a confidently wrong eta with no outward sign.
        "roofline_unavailable": None,
        # PMU memory-system counters (REPORTED-ONLY diagnostics, NOT reward
        # inputs -- see _derive_mem_fields). Whole-program, None unless the
        # timing-mode sim ran and produced a readable --pmu_statsfile.
        "scalar_l1d_miss_rate": None, "l2_miss_rate": None,
        "ddr_read_requests": None, "ddr_write_requests": None,
        "l2fetch_hit_rate": None, "l2fetch_commands": None,
        "vtcm_accesses": None, "hvx_l2_miss": None,
        "pmu_wholeprogram": True,
    }


def _set_mech_from_disasm(fb, dis):
    """Set the static-ELF mechanism flags from one objdump -d text (fail-closed).

    Uses the same candidate-only disasm text as used_hvx/used_hmx, so only
    candidate-authored code is judged.
    """
    fb["used_dma"] = anticheat._disasm_has_dma(dis)
    fb["used_l2fetch"] = anticheat._disasm_has_l2fetch(dis)
    # STATIC half of used_vtcm. Set here rather than deferred to _set_used_vtcm because
    # several paths (sim timeout, and the untimed "skip the timing sim" fast path) return
    # before that runs -- setting it now means untimed runs report VTCM too.
    fb["used_vtcm"] = anticheat._disasm_has_vtcm(dis)


def _set_used_vtcm(fb):
    """VTCM scratchpad was used: STATIC addressing (already set) OR the runtime counter.

    Was PMU-only, which under-counted in two measured ways -- the counter is derived
    from UDMA_VTCM_RD/WR and so only sees VTCM traffic moved by a DMA engine. It misses
    VTCM used as direct HVX scratch (no DMA at all, which is how the HMX helper works),
    and it reads 0 for a single large DMA transfer even though the transfer happens
    (the same bytes split into 4 KB chunks read 480, and removing the DMA makes the
    kernel incorrect, so it was genuinely moving data).

    ORing is deliberately monotone: anything the counter caught before is still caught,
    so this can only ADD detections, never retract one. Static addressing is also the
    house convention -- every other mechanism flag comes from the static ELF -- and it
    is timing-independent, so vtcm no longer depends on timing=True. The counter is
    kept as the second half because an address computed at runtime (rather than named
    as a literal) leaves no static trace. None-safe."""
    fb["used_vtcm"] = bool(fb.get("used_vtcm")) or bool(fb.get("vtcm_accesses"))


# A hexagon-clang compile normally finishes in seconds; a pathological generated
# kernel can drive clang into a multi-hour hang (observed: 254 min at 90% CPU) and
# stall the whole reward/eval loop. Bound it (fail-closed, like the sim timeout).
COMPILE_TIMEOUT_S = 180


def evaluate(candidate_c, task_id=DEFAULT_TASK, sdk_root=DEFAULT_SDK_ROOT, keep=False,
             timing=TIMING_MODE, bus_penalty=BUS_PENALTY, bus_ratio=BUS_RATIO,
             task_dir=None, fast_incorrect=True):
    """Return a feedback dict for one candidate kernel.

    fast_incorrect (default True): when timing=True, run a cheap functional-mode
    sim first to check correctness.  Only correct kernels proceed to the expensive
    timing-mode sim (cycles/eta are only meaningful for correct kernels).  Incorrect
    kernels return immediately with cycles from the fast functional run (reward is
    unchanged: map_reward returns 0.1 for compiled-but-incorrect regardless of cycles).
    Set fast_incorrect=False to always run the full timing sim (original behaviour).
    """
    task_dir = task_dir or os.path.join(TASKS_DIR, task_id)
    harness_c = os.path.join(task_dir, "harness.c")
    fb = _new_feedback(task_id, candidate_c)
    fb["sim_mode"] = "timing" if timing else "functional"
    fb["target"] = _target.current().core if timing else None
    if not os.path.isfile(harness_c):
        fb["error_text"] = f"unknown task '{task_id}': no harness at {harness_c}"
        return fb

    caps = task_caps_in(task_dir)
    cflags = cflags_for_caps(caps)

    sim_timeout = _sim_timeout_for(task_dir)  # size-aware; XL tasks get more than the 60s base

    bin_dir = find_toolchain_bin(sdk_root)
    # C++ (2026-08-03): hexagon-clang++ with -std=c++17 from toolchain.COMPILER.
    # candidate_kernel must be extern "C" or its symbol mangles, breaking both
    # harness linkage and the anti-cheat symbol scoping.
    clang = os.path.join(bin_dir, _exe(COMPILER))
    sim = os.path.join(bin_dir, _exe("hexagon-sim"))

    env = toolchain_env(bin_dir)

    workdir = tempfile.mkdtemp(prefix="hvxenv_")
    elf = os.path.join(workdir, "candidate.elf")

    # --- compile ---
    compile_cmd = [
        clang, *cflags,
        f"-I{task_dir}", f"-I{COMMON_DIR}", "-o", elf, harness_c, os.path.abspath(candidate_c),
    ]
    rc, _, cerr, ctimed = run(compile_cmd, env, timeout=COMPILE_TIMEOUT_S)
    if ctimed or rc != 0 or not os.path.exists(elf):
        fb["error_text"] = (f"compile timed out after {COMPILE_TIMEOUT_S}s" if ctimed
                            else (cerr.strip() or "compilation failed (no diagnostics captured)"))
        fb["timed_out"] = bool(ctimed)
        try:
            os.rmdir(workdir)
        except OSError:
            pass
        return fb
    fb["compiled"] = True

    # --- ELF-level used_hvx (anti-cheat): disassemble the candidate compiled
    # alone, so only candidate-authored code is judged (harness/CRT/libc are a
    # different translation unit and cannot leak HVX credit). Fail-closed.
    objdump = os.path.join(bin_dir, _exe("hexagon-llvm-objdump"))
    obj = os.path.join(workdir, "candidate.o")
    rc_o, _, _, otimed = run(
        [clang, *cflags,
         f"-I{task_dir}", f"-I{COMMON_DIR}", "-c", os.path.abspath(candidate_c),
         "-o", obj],
        env, timeout=COMPILE_TIMEOUT_S,
    )
    dis_obj = ""  # kept for _set_genuine_flags: names the candidate-defined symbols
    if rc_o == 0 and not otimed and os.path.exists(obj):
        rc_d, dis, _, dtimed = run([objdump, "-d", obj], env, timeout=SIM_TIMEOUT_S)
        if rc_d == 0 and dis and not dtimed:
            dis_obj = dis
            fb["used_hvx"] = anticheat._disasm_has_hvx(dis)
            fb["used_hvx_compute"] = anticheat._disasm_has_hvx_compute(dis)
            fb["used_hmx"] = anticheat._disasm_has_hmx(dis)
            _set_mech_from_disasm(fb, dis)  # used_dma / used_l2fetch: static ELF flags
            s_insns, s_pkts, s_hvx = anticheat._disasm_static_counts(dis)
            fb["static_insns"] = s_insns
            fb["static_packets"] = s_pkts
            fb["static_hvx_insns"] = s_hvx
            if s_pkts > 0:
                fb["pack_p_static"] = round(s_insns / s_pkts, 4)
            if s_insns > 0:
                fb["vec_frac"] = round(s_hvx / s_insns, 4)

    # --- simulate ---
    # fast_incorrect: when running in timing mode, use a cheap functional-mode
    # sim first.  Correctness is identical in both modes (same harness, same ELF,
    # same inputs -- only the cycle counter model differs).  If the kernel is
    # incorrect, return immediately; cycles/eta are irrelevant to reward for
    # incorrect kernels (map_reward returns 0.1 regardless of cycles).
    # Only correct kernels pay the cost of the full timing-mode sim.
    use_prefilter = timing and fast_incorrect
    if use_prefilter:
        # Fast functional sim: ~0.5s vs ~10-30s for timing mode.
        fast_cmd = [sim, f"-m{DSP_ARCH}"] + sim_flags_for_caps(caps) + [elf]
        rc_f, sout_f, serr_f, timed_f = run(fast_cmd, env, timeout=sim_timeout)
        if timed_f:
            fb["timed_out"] = True
            fb["error_text"] = f"simulator timed out after {sim_timeout}s"
            _cleanup(workdir, elf, obj, keep, fb)
            return fb
        combined_f = sout_f + "\n" + serr_f
        is_correct = "HVXENV_CORRECT" in combined_f
        if not is_correct:
            # Skip expensive timing sim: populate from functional output and return.
            # sim_mode reflects what actually ran for this path.
            fb["sim_mode"] = "functional"
            fb["target"] = None
            _parse_sim_output(combined_f, fb, timing=False)
            attach_roofline(fb, task_dir)
            _cleanup(workdir, elf, obj, keep, fb)
            return fb
        # Kernel is correct: fall through to timing sim below.

    sim_cmd = [sim, f"-m{DSP_ARCH}"]
    sim_cmd += sim_flags_for_caps(caps)
    pmu_path = None
    pkt_path = None
    if timing:
        pmu_path = os.path.join(workdir, "pmu_stats.txt")
        # --packet_analyze emits per-packet commits/stalls/events as JSON
        # (Simulator UG 80-N2040-1786 Rev. AF §3.9 p45). `commits` is what makes
        # the genuine_* flags possible: it distinguishes an instruction that is
        # present in the ELF from one that actually ran. Timing-mode only, same
        # as --pmu_statsfile.
        pkt_path = os.path.join(workdir, "packet_stats.json")
        sim_cmd += ["--timing",
                    "--buspenalty", str(bus_penalty),
                    "--busratio", str(bus_ratio),
                    "--pmu_statsfile", pmu_path,
                    "--packet_analyze", pkt_path]
    sim_cmd += [elf]
    rc, sout, serr, timed_out = run(sim_cmd, env, timeout=sim_timeout)
    fb["timed_out"] = timed_out
    if timed_out:
        fb["error_text"] = f"simulator timed out after {sim_timeout}s"
        _cleanup(workdir, elf, obj, keep, fb, pmu_path)
        return fb

    combined = sout + "\n" + serr
    _parse_sim_output(combined, fb, timing)
    attach_roofline(fb, task_dir)
    # PMU memory counters: only valid/emitted in timing mode (--pmu_statsfile
    # requires --timing per the sim docs). Fail-open: a missing/garbled file
    # (_read_pmu_stats -> None) just leaves the derived fields at their None
    # defaults -- never raises, never blocks the candidate.
    if timing and pmu_path:
        fb.update(_derive_mem_fields(_read_pmu_stats(pmu_path)))
    _set_used_vtcm(fb)  # RUNTIME flag: vtcm_accesses populated (or not) above
    # RUNTIME mechanism verdicts. Runs AFTER the sim (needs the packet profile)
    # and re-disassembles the LINKED elf (needs runtime addresses). Fail-closed
    # and exception-guarded: a profiling failure must never fail a candidate that
    # compiled and ran, it just leaves genuine_evidence False.
    if timing and pkt_path:
        try:
            _set_genuine_flags(fb, dis_obj, elf, objdump, env, pkt_path)
        except Exception:  # noqa: BLE001 -- diagnostics must never break the reward path
            pass
    _cleanup(workdir, elf, obj, keep, fb, pmu_path, pkt_path)
    return fb


def _set_genuine_flags(fb, dis_obj, elf, objdump, env, pkt_path):
    """RUNTIME mechanism verdicts: present AND executed AND not provably futile.

    The static ``used_*`` flags above answer only "is the instruction present".
    Measured 2026-08-02: a kernel with HVX and l2fetch in a never-taken branch
    sets ``used_hvx``/``used_hvx_compute``/``used_l2fetch`` while executing none
    of it. See ``hexkernels.anticheat.anticheat_runtime``.

    Three steps, because two address spaces are involved:
      1. ``dis_obj`` (candidate.o) names the symbols the candidate DEFINES, but
         its addresses are unrelocated and cannot correlate with the profile.
      2. The linked ELF is disassembled for real addresses, then scoped to those
         symbols so harness/CRT/libc still cannot leak credit.
      3. ``--packet_analyze`` says which packets committed; the frozen detectors
         then run against the executed subset only.

    Work evidence reuses the whole-program PMU counters already read for the
    ``l2fetch_*`` diagnostics (``L2FETCH_ACCESS`` vs the futility counters).

    FAIL-CLOSED, unlike the reported-only PMU diagnostics next door: no profile,
    no disassembly, or no symbols means every ``genuine_*`` stays False and
    ``genuine_evidence`` is False. Absence of proof is never proof of use.
    A functional-mode run therefore reports all-False with evidence False --
    "not demonstrated", not "not present"; the static ``used_*`` flags are
    unaffected and remain the answer to that question.
    """
    if not pkt_path or not os.path.exists(pkt_path):
        return
    try:
        with open(pkt_path, encoding="utf-8", errors="replace") as f:
            profile = _acr.parse_packet_profile(f.read())
    except OSError:
        return
    if not profile:
        return
    executed = _acr.executed_packet_addresses(profile)
    fb["packets_profiled"] = len(profile)
    fb["packets_executed"] = len(executed)

    rc, dis_elf, _, timed = run([objdump, "-d", elf], env, timeout=SIM_TIMEOUT_S)
    if rc != 0 or not dis_elf or timed:
        return
    scoped = _acr.scope_to_symbols(dis_elf, _acr.defined_symbols(dis_obj))
    if not scoped:
        return
    live = _acr.executed_disasm(scoped, executed)
    fb["genuine_evidence"] = True

    totals = _acr.event_totals(profile)
    for key, detector, mech in (
        ("genuine_hvx_compute", anticheat._disasm_has_hvx_compute, "hvx_compute"),
        # Region-scoped, not whole-file: `activation.` in one function and
        # `weight.` in another never compose into a multiply. Tighter than the
        # static used_hmx, which is why the two can legitimately disagree.
        ("genuine_hmx", _acr.has_hmx_matmul_in_one_region, "hmx"),
        ("genuine_dma", anticheat._disasm_has_dma, "dma"),
        ("genuine_l2fetch", anticheat._disasm_has_l2fetch, "l2fetch"),
        ("genuine_vtcm", anticheat._disasm_has_vtcm, "vtcm"),
    ):
        if not detector(live):
            continue
        # None (no counter mapped for this mechanism) is UNKNOWN, not negative --
        # collapsing it to False would reject every genuine HVX/HMX/DMA kernel
        # on the strength of a gap in our counter table.
        fb[key] = _acr.mechanism_did_work(profile, mech) is not False
    if totals:
        fb["l2fetch_futile"] = sum(
            totals.get(c, 0) for c in _acr.FUTILE_COUNTERS["l2fetch"]) or None


def _cleanup(workdir, elf, obj, keep, fb, pmu_path=None, pkt_path=None):
    """Remove build artifacts unless keep=True.

    ``pkt_path`` (the --packet_analyze JSON) matters more than the others: it is
    ~600 KB even for a trivial program, so leaving it behind across a full
    benchmark sweep would be tens of GB.
    """
    if not keep:
        try:
            os.remove(elf)
            if os.path.exists(obj):
                os.remove(obj)
            if pmu_path and os.path.exists(pmu_path):
                os.remove(pmu_path)
            if pkt_path and os.path.exists(pkt_path):
                os.remove(pkt_path)
            os.rmdir(workdir)
        except OSError:
            pass
    else:
        fb["workdir"] = workdir


def summarize(fb):
    if not fb["compiled"]:
        verdict = "DID NOT COMPILE"
    elif fb["timed_out"]:
        verdict = "TIMED OUT"
    elif fb["correct"]:
        verdict = "CORRECT"
    elif fb["errors"] is not None:
        verdict = f"INCORRECT ({fb['errors']} mismatches)"
    else:
        verdict = "NO RESULT"
    hvx = "HVX" if fb["used_hvx"] else "scalar"  # ELF verdict (not source text)
    cyc = fb["cycles"] if fb["cycles"] is not None else "-"
    pp = f"{fb['pack_p_dyn']:.2f}" if fb["pack_p_dyn"] is not None else "-"
    vf = f"{fb['vec_frac']:.2f}" if fb["vec_frac"] is not None else "-"
    return (f"{os.path.basename(fb['candidate']):32s}  {verdict:28s}  elf:{hvx:6s}  "
            f"cycles:{cyc}  pack_p:{pp}  vec_frac:{vf}")


def main():
    ap = argparse.ArgumentParser(description="Headless compile+sim driver (M1).")
    ap.add_argument("candidate", nargs="?", help="path to candidate .c kernel")
    ap.add_argument("--task-id", default=DEFAULT_TASK, help="task id under m1_driver/tasks/")
    ap.add_argument("--sdk-root", default=os.environ.get("HEXAGON_SDK_ROOT", DEFAULT_SDK_ROOT))
    ap.add_argument("--json", action="store_true", help="emit feedback dict as JSON")
    ap.add_argument("--keep", action="store_true", help="keep build artifacts")
    ap.add_argument("--selftest", action="store_true", help="run all sample kernels")
    ap.add_argument("--timing", action=argparse.BooleanOptionalAction, default=TIMING_MODE,
                    help="cycle-approximate timing model (caches+bus); --no-timing idealizes memory")
    ap.add_argument("--buspenalty", type=int, default=BUS_PENALTY)
    ap.add_argument("--busratio", type=int, default=BUS_RATIO)
    args = ap.parse_args()

    ev = lambda k: evaluate(k, task_id=args.task_id, sdk_root=args.sdk_root, keep=args.keep,
                            timing=args.timing, bus_penalty=args.buspenalty,
                            bus_ratio=args.busratio)

    if args.selftest:
        kernels = sorted(glob.glob(os.path.join(HERE, "kernels", "*.c")))
        print(f"Self-test: {len(kernels)} kernels, SDK={args.sdk_root}\n")
        print(f"sim_mode: {'timing (v68n_1024, buspenalty=%d busratio=%d)' % (args.buspenalty, args.busratio) if args.timing else 'functional (memory idealized)'}\n")
        results = []
        for k in kernels:
            fb = ev(k)
            print(summarize(fb))
            results.append(fb)
        if args.json:
            print("\n" + json.dumps(results, indent=2))
        return

    if not args.candidate:
        ap.error("provide a candidate .c file or use --selftest")

    fb = ev(args.candidate)
    if args.json:
        print(json.dumps(fb, indent=2))
    else:
        print(summarize(fb))
        if fb["error_text"]:
            print("\n--- error_text ---\n" + fb["error_text"])


if __name__ == "__main__":
    main()
