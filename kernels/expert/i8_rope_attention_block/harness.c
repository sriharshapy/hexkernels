/* i8_rope_attention_block harness (v5, L3 composition).
 * Full single-head attention with RoPE pre-rotation of Q and K:
 *   RoPE-rotate(Q,K) -> QK^T -> scale+softmax -> A.V, all fixed-point integer.
 * The harness owns main(): fills seeded inputs, builds runtime cos/sin/exp tables,
 * computes an INDEPENDENT scalar golden reference (NOT reusing baseline.c), poisons
 * the output, times the candidate (kernel-only pcycles), does a bit-exact compare.
 * Enables the HMX context AND an identity VTCM translation so a candidate can compose
 * HMX + VTCM + HVX. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>

#define S 64
#define D 64
#define HALF (D/2)

static uint8_t  Q[S*D]      HVX_ALIGN;
static int8_t   K[S*D]      HVX_ALIGN;
static int8_t   V[S*D]      HVX_ALIGN;
static int8_t   cos_lut[S*HALF] HVX_ALIGN;
static int8_t   sin_lut[S*HALF] HVX_ALIGN;
static uint16_t out[S*D]    HVX_ALIGN;
static uint16_t ref[S*D]    HVX_ALIGN;
static uint8_t  exp_lut[256] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF; if (v & 0x800) v -= 0x1000; return v;
}
static inline int clamp_i8(int v) {
    if (v >  127) return  127; if (v < -128) return -128; return v;
}
static inline int clamp_u8(int v) {
    if (v > 255) return 255; if (v < 0) return 0; return v;
}

/* INDEPENDENT scalar golden: RoPE + full attention block, straight from the math
 * (NOT reusing baseline.c) so a buggy shared reference can't hide. */
static void rope_attn_ref(const uint8_t *q, const int8_t *k, const int8_t *v,
                          const int8_t *cs, const int8_t *sn, int rot_shift,
                          uint16_t *r, const uint8_t *lut) {
    static uint8_t qr[S*D];
    static int8_t  kr[S*D];
    static int8_t  scaled[S*S];
    static uint8_t probs[S*S];

    /* RoPE rotation of Q,K over feature pairs (d, d+HALF) */
    for (int i = 0; i < S; i++)
        for (int d = 0; d < HALF; d++) {
            int c = cs[i*HALF+d], s = sn[i*HALF+d];
            int q0 = q[i*D+d], q1 = q[i*D+d+HALF];
            qr[i*D+d]      = (uint8_t)clamp_u8((q0*c - q1*s) >> rot_shift);
            qr[i*D+d+HALF] = (uint8_t)clamp_u8((q0*s + q1*c) >> rot_shift);
            int k0 = k[i*D+d], k1 = k[i*D+d+HALF];
            kr[i*D+d]      = (int8_t)clamp_i8((k0*c - k1*s) >> rot_shift);
            kr[i*D+d+HALF] = (int8_t)clamp_i8((k0*s + k1*c) >> rot_shift);
        }

    /* Stage 1: scores = Qr.Kr^T, HMX 0x40 requant, then scale + clamp to int8 */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)qr[i*D+d] * (int)kr[j*D+d];
            int s12 = hvx_hmx_requant_0x40(acc);
            scaled[i*S+j] = (int8_t)clamp_i8(sx12(s12) >> 4);
        }

    /* Stage 2: row-wise softmax over the KEY axis j (fixed-point LUT) */
    for (int i = 0; i < S; i++) {
        int m = scaled[i*S+0];
        for (int j = 1; j < S; j++) if (scaled[i*S+j] > m) m = scaled[i*S+j];
        int Sr = 0;
        for (int j = 0; j < S; j++) {
            int diff = (int)scaled[i*S+j] - m;
            if (diff < -255) diff = -255;
            uint8_t e = lut[diff + 255];
            probs[i*S+j] = e;
            Sr += (int)e;
        }
        int half = Sr / 2;
        for (int j = 0; j < S; j++)
            probs[i*S+j] = (uint8_t)(((int)probs[i*S+j] * 255 + half) / Sr);
    }

    /* Stage 3: out = probs . V, HMX 0x40 requant */
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
            int acc = 0;
            for (int j = 0; j < S; j++) acc += (int)probs[i*S+j] * (int)v[j*D+d];
            r[i*D+d] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
}

/* Small Q3 sine table (amplitude 8 == "1.0"), for building a RoPE-like rotation. */
static const int8_t SINT[16] = {0,3,6,7,8,7,6,3,0,-3,-6,-7,-8,-7,-6,-3};

int main(void) {
    /* Map VTCM identity so a candidate can stage HMX crouton tiles on-chip. */
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x5AC31D07u;
    const int rot_shift = 4;

    /* Q in 0..7 (positive: int8==uint8). K,V in -3..3.
     * RoPE with cos/sin in [-8,8] and rot_shift=4:
     *   Qr = clamp_u8((q0*c - q1*s)>>4) in [0,7];  Kr = clamp_i8(...>>4) in [-3,3].
     * QK^T: |acc| <= D*7*3 = 1344 -> *17/16 = 1428 < 2048 (12-bit field exact).
     * A.V:  probs is a normalized row (sum ~255), V<=3 -> |acc| <= ~765 -> 813 < 2048.
     * So the whole integer block is BIT-EXACT. */
    for (int i = 0; i < S*D; i++) Q[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < S*D; i++) K[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int i = 0; i < S*D; i++) V[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    /* RoPE edge cases:
     *   position 0 -> identity rotation (cos=8=1.0, sin=0) => Qr[0,:] = Q[0,:]>>4
     *   a 90-degree pair at (i=1,d=0) -> cos=0,sin=8 swaps the pair              */
    /* exp softmax edge: query row 1 all-zero -> uniform softmax after rotation.   */
    for (int d = 0; d < D; d++) Q[1*D+d] = 0;

    /* Build per-position RoPE cos/sin tables from the Q3 sine table (runtime). */
    for (int i = 0; i < S; i++)
        for (int d = 0; d < HALF; d++) {
            int idx = (i * (d + 1)) & 15;
            cos_lut[i*HALF+d] = SINT[(idx + 4) & 15];   /* cos = sin(theta + pi/2) */
            sin_lut[i*HALF+d] = SINT[idx & 15];
        }
    /* Pin the documented edge rotations. */
    cos_lut[0*HALF+0] = 8; sin_lut[0*HALF+0] = 0;   /* identity at (0,0) */
    cos_lut[1*HALF+0] = 0; sin_lut[1*HALF+0] = 8;   /* 90-degree at (1,0) */

    /* Build a real exponential LUT in fixed point (Q16), no libm:
     *   exp_lut[i] = round(255 * exp((i-255)/TAU)),  DECAY_Q16 = round(exp(-1/24)*65536). */
    {
        const uint32_t DECAY_Q16 = 62865u;
        uint64_t vq = (uint64_t)255u << 16;
        for (int i = 255; i >= 0; i--) {
            exp_lut[i] = (uint8_t)((vq + 32768u) >> 16);
            vq = (vq * DECAY_Q16) >> 16;
        }
        exp_lut[255] = 255;
    }

    rope_attn_ref(Q, K, V, cos_lut, sin_lut, rot_shift, ref, exp_lut);

    for (int i = 0; i < S*D; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, {
        candidate_kernel(Q, K, V, cos_lut, sin_lut, out, S, D, rot_shift, exp_lut);
    });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*D; i++) {
        unsigned short g = out[i], e = ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; } }
    }
    hvx_report_u16(errors, S*D, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
