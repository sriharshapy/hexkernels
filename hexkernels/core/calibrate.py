"""Measure the memory hierarchy and emit ``timing_model.json``.

THE GENERATOR THIS REPO NEVER HAD. ``timing_model.json`` arrived in a single
commit (`7c650cd`, "ported") with no derivation anywhere in this repository's
history. Its cache sizes carried no provenance at all, and its bandwidths --
though clearly measured, they have sweep points and r-squared values -- came
from a machine and a script nobody here can point at. That made the whole
roofline unciteable, and it is why retargeting v68 -> v75 could not simply reuse
the file: the numbers describe a different chip's memory system.

METHOD
------
The roofline uses a Hockney model, ``eff_bw(B) = r_inf * B / (B + n_half)``.
Rearranged, the cycles to stream B bytes are linear in B::

    cycles(B) = B / r_inf + n_half / r_inf  =  a + b*B
    =>  r_inf = 1 / b,   n_half = a / b

So the measurement is: stream B bytes, record cycles, fit a straight line.

Fixed overhead (CRT startup, printf, the harness itself) is cancelled exactly
by a DIFFERENTIAL: the same kernel is run at R and 2R iterations and

    cycles_per_pass = (P(2R) - P(R)) / R

Any cost that does not scale with R -- and that is all of the startup cost --
subtracts away. This is why no timer library is needed and why the result does
not depend on how heavy the surrounding program is.

Streaming uses HVX vector loads with an accumulator that escapes, plus a memory
barrier per outer iteration, so the compiler can neither hoist the inner loop
out of the repeat loop nor delete the reads.

LEVELS
------
Which level a sweep exercises is decided by the footprint against the probed
hierarchy: at or under L1-D it is L1D, at or under L2 it is L2, beyond that it
is DDR. VTCM is swept separately by placing the buffer at the target's VTCM
aperture, since VTCM is a software-managed scratchpad rather than part of the
automatic hierarchy.

    python -m core.calibrate --out hexbench/env/timing_model.json

Runs under ``--timing``; a full sweep takes tens of minutes.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

from hexkernels.core import target as _target
from hexkernels.core.toolchain import (BUS_PENALTY, BUS_RATIO, CXX_STD, COMPILER,
                                    DEFAULT_SDK_ROOT, _exe, find_toolchain_bin,
                                    toolchain_env)

# Streaming probe. `-DBYTES` and `-DREPS` are set per point.
#
# The accumulator is written to a volatile sink so the loop cannot be deleted,
# and the barrier stops the compiler hoisting the inner loop out of the repeat
# loop (the buffer never changes, so without it the whole sweep would measure
# one pass).
PROBE = r"""
#include <cstdio>
#include <cstdint>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef BYTES
#define BYTES 4096
#endif
#ifndef REPS
#define REPS 64
#endif

#define VEC 128
#define NVEC (BYTES / VEC)

#if VTCM_BUFFER
static HVX_Vector *const buf = (HVX_Vector *)(uintptr_t)(VTCM_BASE);
#else
alignas(VEC) static unsigned char storage[BYTES];
static HVX_Vector *const buf = (HVX_Vector *)storage;
#endif

volatile int sink;

/* FOUR INDEPENDENT ACCUMULATORS, deliberately.
 *
 * A single accumulator (`acc = vadd(acc, buf[i])`) serialises: every add waits
 * on the previous one, so the loop is LATENCY-bound and tops out around one
 * load per two cycles -- ~64 B/cycle -- no matter which memory it reads.
 * Measured exactly that on the first attempt: L1D 54, L2 64, DDR 64 B/cycle,
 * with DDR indistinguishable from L2 and L1D somehow the slowest. Those were
 * not bandwidth numbers at all.
 *
 * Four independent chains let four loads be in flight, so the loop becomes
 * throughput-bound and the memory system is what limits it. Sanity check: L1D
 * should approach one full 128-byte vector load per cycle. */
int main(void) {
    HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
    HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
    for (int r = 0; r < REPS; ++r) {
        int i = 0;
        for (; i + 3 < NVEC; i += 4) {
            a0 = Q6_Vw_vadd_VwVw(a0, buf[i + 0]);
            a1 = Q6_Vw_vadd_VwVw(a1, buf[i + 1]);
            a2 = Q6_Vw_vadd_VwVw(a2, buf[i + 2]);
            a3 = Q6_Vw_vadd_VwVw(a3, buf[i + 3]);
        }
        for (; i < NVEC; ++i) a0 = Q6_Vw_vadd_VwVw(a0, buf[i]);
        __asm__ __volatile__("" ::: "memory");   /* no hoisting across repeats */
    }
    HVX_Vector acc = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1),
                                     Q6_Vw_vadd_VwVw(a2, a3));
    sink = *(const int *)&acc;
    std::printf("done %d\n", sink);
    return 0;
}
"""

_PCYCLES = re.compile(r"Pcycles\s*[=:]\s*(\d+)", re.I)


def _pcycles(text):
    """Total Pcycles from the simulator's end-of-run summary."""
    hits = _PCYCLES.findall(text)
    return int(hits[-1]) if hits else None


def _run_point(nbytes, reps, vtcm, workdir, binf, env, arch, tgt):
    src = os.path.join(workdir, "probe.cpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(PROBE)
    elf = os.path.join(workdir, f"p_{nbytes}_{reps}_{int(vtcm)}.elf")
    cc = os.path.join(binf, _exe(COMPILER))
    cmd = [cc, f"-m{arch}", "-mhvx", "-mhvx-length=128B", f"-std={CXX_STD}", "-O2",
           f"-DBYTES={nbytes}", f"-DREPS={reps}",
           f"-DVTCM_BUFFER={1 if vtcm else 0}", f"-DVTCM_BASE={tgt.vtcm_base:#x}u",
           "-o", elf, src]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return None
    sim = os.path.join(binf, _exe("hexagon-sim"))
    r = subprocess.run([sim, f"--m{arch}", "--timing",
                        "--buspenalty", str(BUS_PENALTY),
                        "--busratio", str(BUS_RATIO), elf],
                       env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    os.remove(elf)
    return _pcycles(r.stdout + "\n" + r.stderr)


def measure(nbytes, reps, vtcm, workdir, binf, env, arch, tgt):
    """Cycles for ONE pass over `nbytes`, with fixed overhead differenced out."""
    lo = _run_point(nbytes, reps, vtcm, workdir, binf, env, arch, tgt)
    hi = _run_point(nbytes, reps * 2, vtcm, workdir, binf, env, arch, tgt)
    if lo is None or hi is None or hi <= lo:
        return None
    return (hi - lo) / reps


def fit(points):
    """Least-squares cycles = a + b*B over [(bytes, cycles)] -> model fields."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(b for b, _ in points)
    sy = sum(c for _, c in points)
    sxx = sum(b * b for b, _ in points)
    sxy = sum(b * c for b, c in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    if b <= 0:
        return None
    mean = sy / n
    ss_tot = sum((c - mean) ** 2 for _, c in points)
    ss_res = sum((c - (a + b * bb)) ** 2 for bb, c in points)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "r_inf_bytes_per_cycle": round(1.0 / b, 4),
        "half_bytes": round(a / b, 2) if a > 0 else 0.0,
        "fit": {"a_cycles": round(a, 4), "b_cycles_per_byte": round(b, 9),
                "r2": round(r2, 6),
                "sweep_points": [[b_, round(c, 1)] for b_, c in points]},
    }


def sweeps(tgt):
    """(level, sizes, use_vtcm) per memory level, sized against the probed
    hierarchy so each sweep actually lands in the level it names."""
    l1, l2 = tgt.l1d_bytes, tgt.l2_bytes
    return [
        ("L1D", [1 << k for k in range(8, 15) if (1 << k) <= l1], False),
        ("L2", [1 << k for k in range(15, 21) if (1 << k) <= l2], False),
        ("DDR", [1 << k for k in range(21, 25)], False),
        ("VTCM", [1 << k for k in range(8, 21) if (1 << k) <= tgt.vtcm_bytes], True),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "timing_model.json"))
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--sdk-root", default=DEFAULT_SDK_ROOT)
    args = ap.parse_args(argv)

    tgt = _target.current()
    binf = find_toolchain_bin(args.sdk_root)
    env = toolchain_env(binf)
    model = {
        "arch": tgt.core,
        "revid": tgt.revid,
        "level_size_bytes": {"L1D": tgt.l1d_bytes, "L2": tgt.l2_bytes,
                             "VTCM": tgt.vtcm_bytes},
        "bandwidth_model": {},
        "bandwidth_bytes_per_cycle": {},
        "bus_params": {"buspenalty": BUS_PENALTY, "busratio": BUS_RATIO},
        "provenance": {
            "method": "differential Pcycles: (P(2R) - P(R)) / R over an HVX "
                      "streaming loop; linear fit cycles = a + b*B; "
                      "r_inf = 1/b, half_bytes = a/b",
            "generator": "core.calibrate",
            "sim_flags": f"--m{tgt.arch} --timing --buspenalty {BUS_PENALTY} "
                         f"--busratio {BUS_RATIO}",
            "hierarchy_source": "probed from the target configuration table "
                                "(__rdcfg) -- see core.target",
        },
    }

    with tempfile.TemporaryDirectory() as wd:
        for level, sizes, vtcm in sweeps(tgt):
            pts = []
            for nb in sizes:
                c = measure(nb, args.reps, vtcm, wd, binf, env, tgt.arch, tgt)
                if c is not None:
                    pts.append((nb, c))
                    print(f"  {level:5s} {nb:9d} B -> {c:12.1f} cyc "
                          f"({nb / c:7.2f} B/cyc)", flush=True)
            f = fit(pts)
            if f:
                model["bandwidth_model"][level] = f
                model["bandwidth_bytes_per_cycle"][level] = f["r_inf_bytes_per_cycle"]
                print(f"  {level}: r_inf={f['r_inf_bytes_per_cycle']} B/cyc  "
                      f"half={f['half_bytes']} B  r2={f['fit']['r2']}", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
