/* gqa_group_reduce_i8 harness. Grouped-Query-Attention head reduction:
 * H head planes summed down to G groups of HPG=H/G heads each, then
 * requantized (round-half-away-from-zero). Harness owns main() and
 * computes the scalar reference INDEPENDENTLY (same pinned formula). */
#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define H   8
#define G   2
#define HPG 4     /* H / G */
#define M   8
#define N   17
#define MN  (M*N) /* 136 -- NOT a multiple of 128 (tail path) */

static int8_t X[H*MN] HVX_ALIGN;
static int8_t Y[G*MN] HVX_ALIGN;
static int8_t ref[G*MN] HVX_ALIGN;

static int8_t reduce_ref(int32_t acc, int32_t mult, int shift) {
    int64_t r    = (int64_t)acc * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (r >= 0) ? ((r + half) >> shift) : -((-r + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

/* Param sweep: set 0 is exactly the group AVERAGE for HPG=4
 * (scale_mult=1, scale_shift=2 -> divide-by-4 with rounding). Set 1 is a
 * different weighted reduce (scale_mult=3, scale_shift=4 -> *3/16). Both
 * scale_mult values are POSITIVE (sign(q) == sign(acc)). */
static const int32_t MULTS[]  = { 1, 3 };
static const int     SHIFTS[] = { 2, 4 };
#define NSETS 2

int main(void) {
    uint32_t s = 0x6A5A11u;

    /* Random per-head-plane data, full int8 range. */
    for (int i = 0; i < H*MN; i++) X[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases (group 0 = heads 0..3, group 1 = heads 4..7). */
    /* idx=0: all 4 heads of group 0 at +127 -> acc=508 (max positive). */
    for (int t = 0; t < HPG; t++) X[(0*HPG+t)*MN + 0] = 127;
    /* idx=1: all 4 heads of group 0 at -128 -> acc=-512 (max negative). */
    for (int t = 0; t < HPG; t++) X[(0*HPG+t)*MN + 1] = -128;
    /* idx=5: all 4 heads of group 1 at -128 -> acc=-512. */
    for (int t = 0; t < HPG; t++) X[(1*HPG+t)*MN + 5] = -128;
    /* Tail region (idx in [128,136)): idx=130 group0 all +127, idx=135 (last) group1 all +127. */
    for (int t = 0; t < HPG; t++) X[(0*HPG+t)*MN + 130] = 127;
    for (int t = 0; t < HPG; t++) X[(1*HPG+t)*MN + 135] = 127;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        int32_t mult  = MULTS[k];
        int     shift = SHIFTS[k];

        for (int g = 0; g < G; g++) {
            for (int idx = 0; idx < MN; idx++) {
                int32_t acc = 0;
                for (int t = 0; t < HPG; t++)
                    acc += (int32_t)X[(g*HPG + t) * MN + idx];
                ref[g*MN + idx] = reduce_ref(acc, mult, shift);
            }
        }

        for (int i = 0; i < G*MN; i++) Y[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, Y, H, G, M, N, mult, shift); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < G*MN; i++) {
            if (Y[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = k * (G*MN) + i;
                    gotv = (long)Y[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, G*MN*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
