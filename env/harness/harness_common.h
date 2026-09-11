#ifndef HVX_HARNESS_COMMON_H
#define HVX_HARNESS_COMMON_H

/* R&D-ONLY, same as hmx_helpers.h beside it. This file is benchmark harness
 * material -- the VTCM aperture, the 128-byte alignment attribute, the fp16
 * crouton offset, the HMX int8 layout and requant, the tolerance compares -- and
 * a forge v2 kernel that includes it has a weaker provenance chain than that
 * pipeline claims. Forge v2 derives from the generated reference, the schedule
 * annotation and the VENDOR intrinsic headers only.
 *
 * Three layers stop it, because two were not enough: forge v2 compiles no longer
 * put this directory on the include path at all (so the include cannot resolve),
 * this `#error` catches a build that reaches the file some other way, and
 * `forge2.provenance.rd_leaks` catches a COPIED BODY, which neither of the
 * others can see. Seven candidates across batches 2 and 3 had included this
 * header before the path was removed. */
#ifdef FORGE2_BUILD
#error "harness_common.h is R&D-only and must not be used by a forge v2 kernel. Derive from the reference, the schedule, and the vendor headers (<hexagon_types.h>, <hexagon_protos.h>, <hvx_hexagon_protos.h>, <hmx_hexagon_protos.h>). See hexbench/forge2/provenance.py."
#endif

#include <stdio.h>
#include <stdint.h>

#define HVX_ALIGN __attribute__((aligned(128)))

/* ---- kernel-only cycle measurement ----
 * Read the user pcycle counter immediately around the candidate call so the
 * reward sees the KERNEL's cycles, not the (huge, identical) CRT/harness
 * startup. In timing mode this delta is the memory-modeled kernel cost.
 * The standalone runtime's EnablePcycle sets SYSCFG.PCYCLEEN so pcyclelo/hi
 * advance; we also set it defensively (harmless if already on). */
static inline void hvx_enable_pcycle(void) {
    unsigned t;
    __asm__ volatile("%0=syscfg\n\t %0=setbit(%0,#5)\n\t syscfg=%0\n\t isync\n\t"
                     : "=&r"(t));   /* SYSCFG.PCYCLEEN = bit 5 */
}
static inline unsigned long long hvx_rdpcyc(void) {
    unsigned lo, hi;
    __asm__ volatile("%0=pcyclelo\n\t %1=pcyclehi\n\t" : "=r"(lo), "=r"(hi));
    return ((unsigned long long)hi << 32) | lo;
}
/* Time STMT (the candidate call) into OUT (unsigned long long). */
#define HVX_TIME_KERNEL(OUT, STMT) do {                 \
    hvx_enable_pcycle();                                \
    unsigned long long _hvx_c0 = hvx_rdpcyc();          \
    STMT;                                               \
    unsigned long long _hvx_c1 = hvx_rdpcyc();          \
    (OUT) = _hvx_c1 - _hvx_c0;                           \
} while (0)

/* Deterministic LCG so every run uses identical inputs (reproducible eval). */
static inline uint32_t hvx_lcg(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}

/* Single machine-parseable result line (got/exp widened to long for int32 tasks). */
static inline void hvx_report(int errors, int n, int first_bad, long got, long exp) {
    if (errors == 0) {
        printf("HVXENV_CORRECT errors=0 n=%d\n", n);
    } else {
        printf("HVXENV_INCORRECT errors=%d n=%d first_bad=%d got=%ld exp=%ld\n",
               errors, n, first_bad, got, exp);
    }
}
/* ---- v4 capability helpers (HMX / VTCM / FP16). Plain C + GPR asm; harmless
 * for int8 tasks that never call them. Verified in m1_driver/capability/. ---- */

/* VTCM base, selected by the architecture the compiler was invoked with.
 * Scratch for tiling / im2col / crouton packing / gather-dest staging.
 *
 * Each value is read from that target's own configuration table
 * (__rdcfg(__vtcm_base) << 16, offsets from the SDK's hexagon_standalone.h):
 *
 *     v68  0xd840 -> 0xd8400000, VTCM 4096 KB
 *     v73+ 0xd900 -> 0xd9000000, VTCM 8192 KB
 *
 * A single hardcoded base is what made this file wrong on retarget: the v68
 * literal points at unmapped memory on v73/v75/v79, and the anti-cheat's
 * matching aperture regex would have silently matched nothing. Keep this in
 * step with hexbench/env/target.py -- both derive from the same probe. */
#if defined(__HEXAGON_ARCH__) && __HEXAGON_ARCH__ >= 73
#define HVX_VTCM_BASE 0xd9000000u
#else
#define HVX_VTCM_BASE 0xd8400000u
#endif

/* Enable the HMX extension context: SSR.XE (bit 29) + SSR.XA=2 (bits 27:25).
 * The standalone runtime enables only HVX; the harness sets the HMX context
 * (program runs privileged) BEFORE calling the candidate, so a candidate writes
 * only compute code. Call once. */
static inline void hvx_hmx_enable(void) {
    unsigned m = (1u << 29) | (2u << 25);  /* 0x24000000 */
    unsigned t;
    __asm__ volatile("%0=ssr\n\t %0=or(%0,%1)\n\t ssr=%0\n\t isync\n\t"
                     : "=&r"(t) : "r"(m));
}

/* FP16 crouton offset for a 32x32 tile (every two rows transposed); same layout
 * for activation, weight, output. */
static inline int hvx_crouton_off(int r, int c) { return (r / 2) * 64 + c * 2 + (r & 1); }

/* FP16 result line: compare is done by the harness on uint16 bit patterns; this
 * just widens the first-bad bit patterns to long for the standard report line. */
static inline void hvx_report_u16(int errors, int n, int first_bad,
                                  unsigned short got, unsigned short exp) {
    hvx_report(errors, n, first_bad, (long)got, (long)exp);
}

/* ---- int8 HMX matmul layout (decoded 2026-06-12; bit-exact 32x32).
 * Three distinct layouts (activation != weight != output) + a requant. See
 * docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md. ---- */

/* Activation: int8 value in the HIGH byte of an fp16-crouton 16-bit slot.
 * 2048-byte tile (load limit 2047); even/low bytes are dead. */
static inline int hvx_hmx_i8_act_off(int i, int k) { return 2 * hvx_crouton_off(i, k) + 1; }

/* Weight: dense int8, 4-deep packed; 1024-byte tile (load limit 1023). */
static inline int hvx_hmx_i8_wgt_off(int k, int j) { return (k / 4) * 128 + j * 4 + (k % 4); }

/* Output: same crouton slot as fp16, read as uint16 (store :after.uh=acc:2x1). */
static inline int hvx_hmx_i8_out_off(int i, int j) { return hvx_crouton_off(i, j); }

/* Requant at bias-config 0x40 (uniform 0x40 fill): scale 17/16, bias 0,
 * arithmetic-shift floor, result is a 12-bit two's-complement field. Keep
 * |r| < 2048 so the field is exact (no int8 saturation in this config). */
static inline int hvx_hmx_requant_0x40(int acc) { return ((acc * 17 + 8) >> 4) & 0xFFF; }

/* ---- tolerance compare for FLOATING-POINT tasks ----
 * HVX fp16/fp32 arithmetic goes through the non-IEEE qfloat (qf16/qf32) path, so
 * a vectorized HVX float kernel is NOT bit-identical to an IEEE scalar reference
 * (and fp reductions reorder). Float-arithmetic tasks therefore accept a tight
 * abs+rel tolerance instead of bit-exactness (integer + HMX-matmul stay exact).
 * Tolerances are ~1-2 ULP of the dtype; override with -D before include. */
#include <string.h>
#ifndef HVX_FP32_ATOL
#define HVX_FP32_ATOL 1e-4f
#endif
#ifndef HVX_FP32_RTOL
#define HVX_FP32_RTOL 1e-3f
#endif
#ifndef HVX_FP16_ATOL
#define HVX_FP16_ATOL 4e-3f
#endif
#ifndef HVX_FP16_RTOL
#define HVX_FP16_RTOL 8e-3f
#endif

static inline int hvx_close_f32(float g, float e) {
    float d = g - e; if (d < 0) d = -d;
    float ae = e < 0 ? -e : e;
    return d <= HVX_FP32_ATOL + HVX_FP32_RTOL * ae;
}
/* compare two fp16 values given their 16-bit patterns (matches the harness's
 * uint16 read of __fp16 buffers). */
static inline int hvx_close_f16bits(unsigned short gb, unsigned short eb) {
    __fp16 gf, ef;
    memcpy(&gf, &gb, 2); memcpy(&ef, &eb, 2);
    float g = (float)gf, e = (float)ef, d = g - e; if (d < 0) d = -d;
    float ae = e < 0 ? -e : e;
    return d <= HVX_FP16_ATOL + HVX_FP16_RTOL * ae;
}
#endif /* HVX_HARNESS_COMMON_H */
