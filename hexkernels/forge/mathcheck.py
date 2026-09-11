"""Check vector math functions against libm ON THE SIMULATOR, before using them.

    python -m hexkernels.forge.mathcheck --source core.h \
        --check "vlog:logf:0.5:1.5" --check "vatan:atanf:-4:4"

WHY THIS IS A TOOL AND NOT A NOTE IN A COMMENT
----------------------------------------------
Batches 16-25 are 50 kernels over one shared math core, and they were 50/50 correct
on the FIRST simulator pass -- where batches 11-15 each needed several rounds. The
single thing that made the difference was checking the core against libm before
instantiating it 50 times, and the reason it mattered is that the failure mode of
NOT doing it is invisible:

  * The first `vlog2` was a Taylor series in (m - 1). It is 8.9e-2 absolute at the
    top of the mantissa range -- 89 times the harness's tolerance -- and `vlog`,
    `vasinh`, `vacosh` and `vatanh` are all built on it. Without this check that is
    EIGHT unrelated-looking kernel failures, each a plausible candidate for a
    lane-order or scheduling bug, and the shared cause is invisible in all eight.
    With it, it is one number on one line.
  * `vatan` with a single range reduction is 5.0e-3 relative at t = 0.9, five times
    the budget, and `vasin`/`vacos` inherited it at 8.6e-3.

Both were found in minutes and fixed once. That is the entire argument for this
file: verify a shared primitive at the point where a failure names itself.

AND IT CAUGHT A BUG IN ITSELF FIRST. The original scratch version declared
`N = 64` floats and computed one vector -- an HVX vector is 32 fp32 lanes -- so
half the array was never written and EVERY function including `vtrunc`, which is
exact, reported over budget, always with its first error at index 32. The harness
being wrong looked exactly like sixteen broken functions. `--lanes` is derived from
the element width here rather than typed in, and `forge2.lint`'s
`vector-lane-width` rule exists because of this.

WHAT IT REPORTS. Max absolute and max relative error over `--samples` points
linearly spanning the range, against the double-precision libm value, beside the
harness's own budget (1e-4 absolute + 1e-3 relative for fp32). Values libm returns
as non-finite are skipped -- an out-of-domain point says nothing about the
approximation, and counting it would make a correct function look broken.
"""
import argparse
import os
import subprocess
import sys
import tempfile

from hexkernels.core import target as _target
from hexkernels.core.toolchain import (COMPILER, CXX_STD, DEFAULT_SDK_ROOT,
                                    find_toolchain_bin, sim_flags_for_caps,
                                    toolchain_env)

#: fp32 only for now: every function in the shared core takes and returns
#: `HVX_Vector` holding fp32 lanes. A fp16 variant would need a different lane
#: count and a different tolerance, and inventing that before something needs it
#: would be guessing at an interface.
LANES = 32
ATOL, RTOL = 1e-4, 1e-3

PRELUDE = r"""
#include <stdio.h>
#include <math.h>

/* LANES IS DERIVED, NOT TYPED. A 128-byte HVX vector holds 128/sizeof(float) = 32
 * fp32 lanes. The first version of this harness hardcoded 64, computed one vector,
 * and reported every function -- including the exact ones -- as over budget with
 * its first error always at index 32. */
#define LANES (128 / (int)sizeof(float))

static float mc_in[LANES]  __attribute__((aligned(128)));
static float mc_out[LANES] __attribute__((aligned(128)));

typedef HVX_Vector (*mc_fn)(HVX_Vector);

static void mc_check(const char *name, mc_fn f, double (*ref)(double),
                     double lo, double hi, int samples)
{
    double worst_abs = 0.0, worst_rel = 0.0, at_abs = lo, at_rel = lo;
    int checked = 0, skipped = 0;

    for (int base = 0; base < samples; base += LANES) {
        for (int i = 0; i < LANES; i++) {
            int k = base + i;
            if (k >= samples) k = samples - 1;
            mc_in[i] = (float)(lo + (hi - lo) * ((double)k / (double)(samples - 1)));
        }
        *(HVX_Vector *)mc_out = f(*(const HVX_Vector *)mc_in);
        for (int i = 0; i < LANES && base + i < samples; i++) {
            double want = ref((double)mc_in[i]);
            /* An out-of-domain point says nothing about the approximation, and
             * counting it would make a correct function look broken. */
            if (!isfinite(want)) { skipped++; continue; }
            double d = fabs((double)mc_out[i] - want);
            double r = d / (fabs(want) > 1e-30 ? fabs(want) : 1.0);
            if (d > worst_abs) { worst_abs = d; at_abs = (double)mc_in[i]; }
            if (r > worst_rel) { worst_rel = r; at_rel = (double)mc_in[i]; }
            checked++;
        }
    }
    int ok = (worst_abs <= %ATOL% || worst_rel <= %RTOL%);
    printf("MATHCHECK %s range=[%g,%g] n=%d skipped=%d max_abs=%.3e at=%g "
           "max_rel=%.3e at=%g %s\n",
           name, lo, hi, checked, skipped, worst_abs, at_abs, worst_rel, at_rel,
           ok ? "ok" : "OVER-BUDGET");
}
"""


def harness(source: str, checks) -> str:
    """`source` (the functions under test) plus a `main` that checks each one.

    `source` is inserted verbatim and must already include whatever it needs --
    the point is to test the EXACT text a kernel will carry, not a transcription
    of it.
    """
    body = [PRELUDE.replace("%ATOL%", repr(ATOL)).replace("%RTOL%", repr(RTOL))]
    for fn, ref, _lo, _hi in checks:
        # A REFERENCE IS OFTEN A COMPOSITION, NOT A LIBM NAME. libm has no `expit`
        # or `softplus`, and checking `vsigmoid` against `expf` reported it 3.7
        # ABSOLUTE off -- a correct function failing because the comparison was
        # wrong, which is the most expensive kind of false alarm this tool can
        # produce. So a `ref` containing a parenthesis is treated as a C expression
        # in `x`; anything else is a function name and is called.
        expr = ref if "(" in ref else f"{ref}(x)"
        body.append(f"static double mc_ref_{fn}(double x) {{ return {expr}; }}")
    body.append("int main(void) {")
    for fn, ref, lo, hi in checks:
        body.append(f"    mc_check(\"{fn}\", {fn}, mc_ref_{fn}, "
                    f"{lo!r}, {hi!r}, SAMPLES);")
    body.append('    printf("MATHCHECK done\\n");')
    body.append("    return 0;")
    body.append("}")
    return source.rstrip() + "\n" + "\n".join(body) + "\n"


def _run_sim(elf, binf, env, timeout):
    """Run `elf` under the simulator, functional mode.

    `sim_flags_for_caps` rather than a spelled-out flag list: HVX and HMX need
    matching flags on BOTH the compile and the simulator, and when those two lists
    were maintained separately the sim half was missed -- which made HMX
    unreachable through the harness no matter what a kernel wrote, and looked like
    the kernel's fault. No `--timing`: this measures ACCURACY, and a cycle count
    would not change a single digit of the answer.
    """
    from hexkernels.forge.verify import _exe, _run
    tgt = _target.current()
    sim = [os.path.join(binf, _exe("hexagon-sim")), f"--m{tgt.arch}"]
    sim += sim_flags_for_caps(["hmx"])
    sim.append(elf)
    return _run(sim, env, timeout)


def check(source: str, checks, *, samples=4096, sdk_root=DEFAULT_SDK_ROOT,
          timeout=900) -> list:
    """Compile and run the checks; return one dict per function."""
    tgt = _target.current()
    binf = find_toolchain_bin(sdk_root)
    env = toolchain_env(binf)
    cc = os.path.join(binf, COMPILER + (".exe" if os.name == "nt" else ""))
    src = harness(source, checks).replace("SAMPLES", str(samples))

    with tempfile.TemporaryDirectory(prefix="mathcheck_") as tmp:
        cpath = os.path.join(tmp, "mathcheck.cpp")
        elf = os.path.join(tmp, "mathcheck.elf")
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(src)
        cmd = [cc, f"-m{tgt.arch}", "-mhvx", f"-mhvx-length={tgt.hvx_bytes}B",
               "-mhmx", f"-std={CXX_STD}", "-O2", "-o", elf, cpath]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError("mathcheck failed to compile:\n"
                               + (r.stderr or r.stdout)[:4000])
        out = _run_sim(elf, binf, env, timeout).stdout

    rows = []
    for line in out.splitlines():
        if not line.startswith("MATHCHECK ") or line.endswith("done"):
            continue
        parts = line.split()
        row = {"name": parts[1], "ok": parts[-1] == "ok"}
        for p in parts[2:-1]:
            if "=" in p:
                k, v = p.split("=", 1)
                row.setdefault(k, v)
        rows.append(row)
    # FAIL CLOSED. No output means the program did not reach its checks -- a fault
    # or a timeout -- and reporting "0 over budget" for that would be the same
    # mistake as a harness that passes when it never ran.
    if "MATHCHECK done" not in out:
        raise RuntimeError("mathcheck did not run to completion:\n" + out[-2000:])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="file holding the functions under test, inserted verbatim")
    ap.add_argument("--check", action="append", required=True, metavar="SPEC",
                    help="fn:ref:lo:hi -- `ref` is a libm name (vlog:logf:0.5:1.5) "
                         "or a C expression in x when it contains a paren "
                         '(vsigmoid:1.0/(1.0+exp(-x)):0.5:1.5)')
    ap.add_argument("--samples", type=int, default=4096)
    args = ap.parse_args(argv)

    checks = []
    for spec in args.check:
        # Split the ENDS off, not on every colon: an expression reference may
        # contain one (a ternary), and `fn:ref:lo:hi` is unambiguous from the edges.
        fn, rest = spec.split(":", 1)
        ref, lo, hi = rest.rsplit(":", 2)
        checks.append((fn, ref, float(lo), float(hi)))

    with open(args.source, encoding="utf-8") as f:
        source = f.read()

    rows = check(source, checks, samples=args.samples)
    width = max(len(r["name"]) for r in rows)
    bad = 0
    for r in rows:
        bad += not r["ok"]
        print(f"  {r['name']:{width}s}  range={r.get('range','?'):18s} "
              f"max_abs={r.get('max_abs','?'):>9s}  "
              f"max_rel={r.get('max_rel','?'):>9s}  "
              f"{'ok' if r['ok'] else 'OVER-BUDGET'}")
    print(f"\n{len(rows) - bad}/{len(rows)} within the harness's budget "
          f"({ATOL:g} absolute OR {RTOL:g} relative)")
    if bad:
        print("\nAn OVER-BUDGET primitive will fail every kernel built on it, and "
              "the failures will not look related. Fix it before instantiating.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
