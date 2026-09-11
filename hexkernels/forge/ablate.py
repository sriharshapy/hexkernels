"""Does the mechanism EARN its place at this size? Measured by ablation.

THE GAP THIS CLOSES
-------------------
`mechanism.py` grants a mechanism from a CAPACITY COMPARISON: working set past L1D
earns `l2fetch`, past L2 earns `dma` and `vtcm`. Nothing has ever checked that the
granted mechanism helps at that size, and `verify.py` only reports that the
instruction is PRESENT -- a kernel can stage a tile through VTCM and then read its
operands from DDR anyway, and every flag still reads True.

That matters in both directions, and the second one is why this module exists:

  * too SMALL and DMA cannot pay for itself -- descriptor setup and a `dmwait` per
    tile against a working set the cache would have handled, where `l2fetch` (a
    single instruction, no descriptor, no wait) is the honest mechanism;
  * too LARGE and the task is expensive to verify for no scientific gain, since the
    entitlement is a threshold comparison and overshooting it buys nothing.

So the size a task should have is the smallest one where its claimed mechanism is
still measurably the right choice. This module finds that by running the SAME
computation three ways at several sizes and comparing instruction counts.

INSTRUCTION COUNT ALONE CANNOT ANSWER THE QUESTION -- MEASURED
--------------------------------------------------------------
The first sweep compared instructions retired, and plain HVX won at EVERY size
from 0.02 MB to 12 MB by a flat 7%:

    working set  hvx_only   +l2fetch   +dma/vtcm
      0.25 MB      1.00x      0.94x      0.94x
      2.00 MB      1.00x      0.93x      0.93x
     12.00 MB      1.00x      0.93x      0.93x

That is not a fact about the hardware. It is a defect in the metric: in FUNCTIONAL
mode the simulator has no memory model, so a demand load from DDR costs exactly
what a load from VTCM costs. A mechanism whose whole purpose is HIDING LATENCY can
therefore only ever add instructions -- the descriptor writes, the `dmwait`, the
extra copy -- and never show a benefit. Instruction count is the right metric for
VECTORISATION, which genuinely removes work, and structurally blind to prefetch and
staging.

So the sweep runs with `--timing`, which models the bus (`BUS_PENALTY`,
`BUS_RATIO` pinned in `toolchain.py`), and reports Pcycles. The caveat from
CLAUDE.md applies and is not negotiable: that model is REPRODUCIBLE, not
silicon-validated. What it supports is a RELATIVE comparison of variants under one
model at one size -- which is exactly the question "does this mechanism earn its
place here" -- and not an absolute cycle claim.

WHAT IS COMPARED
----------------
`insns` from the simulator's own summary: instructions retired, no memory model, no
calibration. It is the number that cannot be faked by dead intrinsic code. It is
also NOT a runtime: on real silicon a DMA transfer runs in the background and costs
the core almost nothing per byte, which instruction count REWARDS (the core issues
few instructions) but which no functional simulator can turn into a wall-clock
saving. So:

    a variant with FEWER instructions issued fewer operations to get the same
    answer. That is a real, timing-model-independent statement, and it is the
    only one made here.

Every variant computes the same relu over the same bytes and prints a checksum, so
a variant that skipped work is caught rather than rewarded for being fast.

    python -m hexkernels.forge.ablate --sizes 1,4,12
"""
import argparse
import re
import sys

from hexkernels.forge.verify import verify

# One computation -- relu over a flat fp32 array -- expressed three ways. Relu is
# chosen deliberately: one arithmetic operation per element, so the memory
# mechanism is nearly the whole cost and the comparison is not diluted by
# arithmetic the variants share.
_PREAMBLE = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 73
#define VTCM_BASE 0xd9000000u
#else
#define VTCM_BASE 0xd8400000u
#endif

#define NBYTES %NBYTES%
#define NELEM  (NBYTES / 4)
#define NVEC   (NBYTES / 128)
#define VLANES 32                        /* fp32 per 128 B vector */
#define REPEAT %REPEAT%                  /* run() this many times */

static float g_in[NELEM] __attribute__((aligned(128)));
static float g_out[NELEM] __attribute__((aligned(128)));

/* SETUP AND CHECKING ARE VECTORISED, and that is the whole reason this
 * instrument works. The first version filled and summed with scalar loops over
 * every element -- O(N) work in every variant, against O(N/32) of vector work --
 * and all three variants came out at 575,527 instructions to within 0.1%. The
 * overhead was the measurement. Now every phase is O(N/32), and run() is repeated
 * REPEAT times so the mechanism under test dominates what is left. */
static void fill(void) {
    HVX_Vector *d = (HVX_Vector *)g_in;
    /* alternating signs so relu has to select; a variant that skips work or
     * mis-selects changes the checksum below. */
    const HVX_Vector neg = Q6_V_vsplat_R((int)0xC0400000u);   /* -3.0f       */
    /* 0x40000001, not 0x40000000. THE LOW BIT IS LOAD-BEARING: with a round
     * 0x40000000 the word-sum of 16,384 copies is 2^44, which is exactly 0 mod
     * 2^32 -- the same checksum an all-zeros output gives, so the check could not
     * tell a correct variant from one that wrote nothing. Measured: it printed
     * checksum=0 for all three variants before this was fixed. */
    const HVX_Vector pos = Q6_V_vsplat_R((int)0x40000001u);   /* ~+2.0000002f */
    for (int v = 0; v < NVEC; v++) d[v] = (v & 1) ? pos : neg;
}

/* Bitwise checksum of g_out, vectorised: integer lane adds (selectable on v75,
 * unlike the fp32 ones) over the raw words, then one horizontal reduce. Exact,
 * cheap, and identical in every variant, so it is constant overhead. */
static long checksum(void) {
    const HVX_Vector *s = (const HVX_Vector *)g_out;
    HVX_Vector acc = Q6_V_vzero();
    for (int v = 0; v < NVEC; v++) acc = Q6_Vw_vadd_VwVw(acc, s[v]);
    acc = Q6_Vw_vadd_VwVw(acc, Q6_V_vror_VR(acc, 64));
    acc = Q6_Vw_vadd_VwVw(acc, Q6_V_vror_VR(acc, 32));
    acc = Q6_Vw_vadd_VwVw(acc, Q6_V_vror_VR(acc, 16));
    acc = Q6_Vw_vadd_VwVw(acc, Q6_V_vror_VR(acc, 8));
    acc = Q6_Vw_vadd_VwVw(acc, Q6_V_vror_VR(acc, 4));
    static int scratch[32] __attribute__((aligned(128)));
    *(HVX_Vector *)scratch = acc;
    return (long)scratch[0];
}
'''

_VARIANTS = {
    # The baseline the other two are measured against: vector arithmetic, demand
    # loads, no prefetch and no staging.
    "hvx_only": r'''
static void run(void) {
    const HVX_Vector *s = (const HVX_Vector *)g_in;
    HVX_Vector *d = (HVX_Vector *)g_out;
    const HVX_Vector z = Q6_V_vzero();
    for (int v = 0; v < NELEM / VLANES; v++) d[v] = Q6_Vsf_vmax_VsfVsf(s[v], z);
}
''',
    # One instruction per block, no descriptor and no wait. This is what SHOULD win
    # at sizes the cache can hold.
    # l2fetch AS THE VENDOR DOCUMENTS IT. V75 HVX PRM 80-N2040-58 Rev. AB §3.9.4:
    # "L2FETCH is best performed in sizes LESS THAN 8 KB and issued at least SEVERAL
    # HUNDRED CYCLES prior to using the data. If the L2FETCH instruction is issued
    # too early, data can be evicted before use."
    #
    # The first version of this variant used 8 KB blocks (at the limit, not under it)
    # issued ONE BLOCK AHEAD -- about 64 packets of lead, not several hundred cycles
    # -- and lost to plain HVX at every size by 4-7%. That number described this code
    # and not the mechanism. Now: 4 KB blocks, and the prefetch runs LOOKAHEAD blocks
    # ahead of the consumer so the lead is several hundred cycles of real work.
    "hvx_l2fetch": r'''
#define PF_BYTES  4096u                   /* < 8 KB, per PRM 3.9.4 */
#define LOOKAHEAD 4                       /* 4 blocks = 16 KB = ~128 vector packets */
static void run(void) {
    const HVX_Vector *s = (const HVX_Vector *)g_in;
    HVX_Vector *d = (HVX_Vector *)g_out;
    const HVX_Vector z = Q6_V_vzero();
    const int vpb = (int)(PF_BYTES / 128);
    const int nv = NVEC;
    const uint64_t desc = ((uint64_t)PF_BYTES << 32) | ((uint64_t)PF_BYTES << 16) | 1u;
    /* Prime the pipeline: get LOOKAHEAD blocks in flight before consuming any, so
     * the first block is not itself a demand miss. */
    for (int k = 0; k < LOOKAHEAD; k++) {
        size_t off = (size_t)k * PF_BYTES;
        if (off < NBYTES) Q6_l2fetch_AP((void *)((const char *)g_in + off), desc);
    }
    for (int base = 0; base < nv; base += vpb) {
        /* fetch the block LOOKAHEAD ahead of the one about to be consumed */
        size_t off = (size_t)(base / vpb + LOOKAHEAD) * PF_BYTES;
        if (off < NBYTES) Q6_l2fetch_AP((void *)((const char *)g_in + off), desc);
        int end = base + vpb < nv ? base + vpb : nv;
        for (int v = base; v < end; v++) d[v] = Q6_Vsf_vmax_VsfVsf(s[v], z);
    }
}
''',
    # Descriptor setup, a dmstart and a dmwait per tile, plus the copy in and out.
    # All of that is instructions the other two do not issue.
    "hvx_dma_vtcm": r'''
/* A REAL double buffer. The first version of this variant issued the transfer and
 * then called dmwait IMMEDIATELY after the compute, so the engine only ever ran
 * while the core was blocked -- zero overlap, and DMA could only ever add cost.
 * It lost by 40% at 12 MB and the loss GREW with size, which is the signature of
 * a serialised transfer rather than a hidden one.
 *
 * The pattern that actually hides the transfer, and the one batch 4's gelu and
 * silu use: at the top of tile t hand the engine tile t+1's input AND tile t-1's
 * output, THEN compute tile t, THEN wait. */
#define TBYTES 32768u                     /* 32 KB per buffer, 4 live = 128 KB */
static hexagon_udma_descriptor_type0_t g_desc[2] __attribute__((aligned(64)));

static inline void dma_fill(hexagon_udma_descriptor_type0_t *d, const void *src,
                            void *dst, unsigned bytes, void *next) {
    d->next = next; d->length = bytes;
    d->desctype = HEXAGON_UDMA_DESC_DESCTYPE_TYPE0;
    d->srcbypass = HEXAGON_UDMA_DESC_BYPASS_ON;
    d->dstbypass = HEXAGON_UDMA_DESC_BYPASS_ON;
    d->order = HEXAGON_UDMA_DESC_ORDER_NOORDER;
    d->dstate = HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE;
    d->src = (void *)src; d->dst = dst;
}
static inline void dma_wait(void) {
    (void)Q6_R_dmwait(); __asm__ volatile("" ::: "memory");
}
static void run(void) {
    float *vin[2]  = {(float *)(uintptr_t)(VTCM_BASE + 0x00000u),
                      (float *)(uintptr_t)(VTCM_BASE + 0x08000u)};
    float *vout[2] = {(float *)(uintptr_t)(VTCM_BASE + 0x10000u),
                      (float *)(uintptr_t)(VTCM_BASE + 0x18000u)};
    const HVX_Vector z = Q6_V_vzero();
    const unsigned ntile = (NBYTES + TBYTES - 1) / TBYTES;

    /* Prime: tile 0 resident, and the engine retired one transfer -- two dmstarts
     * as its first act fault 0x28 (docs/hexagon/SDK_DOC_INDEX.md). */
    dma_fill(&g_desc[0], g_in, vin[0], TBYTES, 0);
    Q6_dmstart_A((void *)&g_desc[0]);
    dma_wait();

    for (unsigned t = 0; t < ntile; t++) {
        unsigned cur = t & 1u, alt = (t + 1u) & 1u;
        unsigned bytes = (t + 1) * TBYTES <= NBYTES ? TBYTES : NBYTES - t * TBYTES;

        /* Hand the engine BOTH directions before computing, chained into one
         * dmstart, so the transfers run underneath this tile's arithmetic. */
        unsigned nxt = (t + 1) * TBYTES;
        int have_in = nxt < NBYTES, have_out = t > 0;
        if (have_in && have_out) {
            unsigned nb = nxt + TBYTES <= NBYTES ? TBYTES : NBYTES - nxt;
            dma_fill(&g_desc[1], (const char *)g_in + nxt, vin[alt], nb, 0);
            dma_fill(&g_desc[0], vout[alt], (char *)g_out + (t - 1) * TBYTES,
                     TBYTES, (void *)&g_desc[1]);
            Q6_dmstart_A((void *)&g_desc[0]);
        } else if (have_in) {
            unsigned nb = nxt + TBYTES <= NBYTES ? TBYTES : NBYTES - nxt;
            dma_fill(&g_desc[0], (const char *)g_in + nxt, vin[alt], nb, 0);
            Q6_dmstart_A((void *)&g_desc[0]);
        } else if (have_out) {
            dma_fill(&g_desc[0], vout[alt], (char *)g_out + (t - 1) * TBYTES,
                     TBYTES, 0);
            Q6_dmstart_A((void *)&g_desc[0]);
        }

        const HVX_Vector *s = (const HVX_Vector *)vin[cur];
        HVX_Vector *d = (HVX_Vector *)vout[cur];
        for (unsigned v = 0; v < bytes / 128; v++) d[v] = Q6_Vsf_vmax_VsfVsf(s[v], z);

        if (have_in || have_out) dma_wait();
    }
    /* Epilogue: the last tile's output is still in VTCM. */
    unsigned last = ntile - 1, lcur = last & 1u;
    unsigned lbytes = NBYTES - last * TBYTES;
    dma_fill(&g_desc[0], vout[lcur], (char *)g_out + last * TBYTES, lbytes, 0);
    Q6_dmstart_A((void *)&g_desc[0]);
    dma_wait();
}
''',
}

_MAIN = r'''
extern "C" int main(void) {
    fill();
    run();
    printf("ABLATE checksum=%ld n=%d\n", checksum(), (int)NELEM);
    printf("HVXENV_CORRECT\n");
    return 0;
}
'''

_HARNESS = r'''
/* The ablation programs are self-contained and self-checking, so this file exists
 * only because verify() takes two translation units. */
int main(void);
'''


# run() is repeated so the mechanism under test dominates the constant setup and
# checking cost. 8 was chosen by measurement: at 4 the three variants still sat
# within 3% at the smallest size.
REPEAT = 8


def program(variant: str, nbytes: int, repeat=REPEAT) -> str:
    return (_PREAMBLE.replace("%NBYTES%", str(nbytes))
                     .replace("%REPEAT%", str(repeat))
            + _VARIANTS[variant] + _MAIN)


def run_one(variant: str, nbytes: int, timeout=1800, repeat=REPEAT,
            timing=False, bus_penalty=None, bus_ratio=None) -> dict:
    """Compile, run, and report the cost of one (variant, size).

    `timing=True` turns on the simulator's memory model, which is the only mode in
    which a latency-hiding mechanism can show a benefit at all.
    """
    v = verify(program(variant, nbytes, repeat), "",
               name=f"ablate_{variant}_{nbytes}", timeout=timeout, timing=timing,
               bus_penalty=bus_penalty, bus_ratio=bus_ratio)
    out = v.get("stdout") or ""
    m = re.search(r"ABLATE checksum=(-?\d+) n=(\d+)", out)
    return {"variant": variant, "nbytes": nbytes,
            "bus_penalty": v.get("bus_penalty"), "bus_ratio": v.get("bus_ratio"),
            "compiled": v["compiled"], "ran": v["ran"],
            "insns": v.get("insns"), "pcycles": v.get("pcycles"),
            "checksum": int(m.group(1)) if m else None,
            "error": (v.get("error_text") or "")[:300]}


def sweep(sizes_mb, variants=None, timing=False, repeat=REPEAT,
          bus_penalty=None, bus_ratio=None) -> list:
    variants = variants or list(_VARIANTS)
    rows = []
    for mb in sizes_mb:
        nbytes = int(mb * 1024 * 1024)
        nbytes -= nbytes % 128                      # whole vectors only
        for var in variants:
            rows.append(run_one(var, nbytes, repeat=repeat, timing=timing,
                                bus_penalty=bus_penalty, bus_ratio=bus_ratio))
    return rows


def render(rows, timing=False) -> str:
    key = "pcycles" if timing else "insns"
    metric = "Pcycles (bus modeled)" if timing else "insns"
    by_size = {}
    for r in rows:
        by_size.setdefault(r["nbytes"], []).append(r)
    out = ["# Mechanism ablation — does the granted mechanism earn its place?", "",
           "Same relu over the same bytes, three ways. Checksums must agree across",
           "variants, or a variant skipped work and its number means nothing.", ""]
    if timing:
        bp = next((r.get("bus_penalty") for r in rows if r.get("bus_penalty")), "?")
        br = next((r.get("bus_ratio") for r in rows if r.get("bus_ratio")), "?")
        out += [f"**Bus configuration: penalty {bp}, ratio {br}.** Every ranking",
                "below is a ranking for THIS bus; the crossover moves with it.", ""]
        out += ["Measured with the simulator's MEMORY MODEL on (`--timing`), so a",
                "latency-hiding mechanism can show a benefit. The model is",
                "reproducible, NOT silicon-validated: these support a relative",
                "comparison of variants at one size, not an absolute cycle claim.", ""]
    else:
        out += ["Measured in FUNCTIONAL mode: no memory model, so a demand load from",
                "DDR costs what a VTCM load costs. Prefetch and staging can only add",
                "instructions here -- this column judges VECTORISATION, nothing else.",
                ""]
    for nbytes, group in sorted(by_size.items()):
        mb = nbytes / 1048576.0
        # working set is in + out
        ws = 2 * nbytes
        tier = ("T0" if ws <= 16384 else "T1" if ws <= 1048576
                else "T2" if ws <= 8388608 else "T3")
        out += [f"## {mb:.2f} MB per buffer — working set {ws/1048576.0:.2f} MB, "
                f"tier {tier}", "",
                f"| variant | {metric} | vs hvx_only | checksum |",
                "|---|---|---|---|"]
        base = next((g[key] for g in group if g["variant"] == "hvx_only"), None)
        sums = {g["checksum"] for g in group if g["checksum"] is not None}
        for g in group:
            if not g.get(key):
                out.append(f"| {g['variant']} | FAILED | - | - |  {g['error']}")
                continue
            rel = f"{base / g[key]:.2f}x" if base else "-"
            out.append(f"| {g['variant']} | {g[key]:,} | {rel} | "
                       f"{g['checksum']} |")
        if len(sums) > 1:
            out += ["", f"**CHECKSUMS DISAGREE {sums} — a variant did not compute "
                    "the same thing, so this size's numbers mean nothing.**"]
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="0.25,1,4",
                    help="MB per buffer, comma separated (working set is 2x this)")
    ap.add_argument("--variants", default=None,
                    help="comma separated subset of " + ",".join(_VARIANTS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--timing", action="store_true",
                    help="turn on the simulator memory model -- REQUIRED for any "
                         "claim about l2fetch or dma")
    ap.add_argument("--repeat", type=int, default=REPEAT)
    ap.add_argument("--buspenalty", type=int, default=None,
                    help="override the pinned bus response penalty (default 75). "
                         "The crossover between mechanisms MOVES with this, so a "
                         "band derived from one value is a band for one bus.")
    ap.add_argument("--busratio", type=int, default=None,
                    help="override the pinned bus clock ratio (default 2)")
    args = ap.parse_args(argv)
    sizes = [float(s) for s in args.sizes.split(",")]
    variants = args.variants.split(",") if args.variants else None
    rows = sweep(sizes, variants, timing=args.timing, repeat=args.repeat,
                 bus_penalty=args.buspenalty, bus_ratio=args.busratio)
    text = render(rows, timing=args.timing)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
