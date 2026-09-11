/* HMX int8 32x32 matmul + requantize-to-int8 (bit-exact) — the ACCELERATED expert.
 * Single 32x32 crouton tile (the proven 2026-06-12 int8 layout): activation lives
 * in the HIGH byte of an fp16-crouton slot (2048B tile, load limit 2047); weight is
 * 4-deep packed (1024B tile, load limit 1023) -- activation and weight need
 * DIFFERENT load limits in one packet -> explicit asm. The 0x40-config epilogue
 * computes r=(acc*17+8)>>4 as a 12-bit two's-complement field zero-extended into a
 * uint16; this task narrows that (sign-extended) field straight down to int8 --
 * the harness picks input ranges so |r|<=127, so the narrow is exact.
 *
 * PERF: a scalar VTCM access costs ~48 cycles in timing mode, so pack/unpack is
 * staged through cacheable buffers and moved to/from VTCM in bulk 128B vector
 * copies (Task-4 lesson: naive scalar VTCM access LOSES to the HVX baseline;
 * staged copies turn it into a real win).
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int8_t hmx_i8_narrow(uint16_t field12) {
    int v = (int)(field12 & 0xFFFu);
    if (v & 0x800) v -= 0x1000;   /* sign-extend the 12-bit two's-complement field */
    return (int8_t)v;             /* exact: caller guarantees |v| <= 127 */
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u); /* 2048B */
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u); /* 1024B */
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;   /* cacheable crouton staging */
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;    /* even/low bytes stay 0 */
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40; /* requant: scale 17/16, bias 0 */

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) aAct[hvx_hmx_i8_act_off(i, k)] = A[i*n + k];
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) aWgt[hvx_hmx_i8_wgt_off(k, j)] = B[k*n + j];

    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");

    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[i*n + j] = hmx_i8_narrow(aOut[hvx_hmx_i8_out_off(i, j)]);
}
