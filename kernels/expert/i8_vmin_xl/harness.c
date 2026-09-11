/* i8_vmin_xl harness — DDR-bound variant (N=500K, working set ~1.5MB > 1MB L2).
 * Uses HVX for init/ref/verify loops so the total simulated cycles stay well under
 * the 60-second simulator wall-clock budget even at large N.
 *
 * Correctness contract:
 *   - SIGNED int8 min: out[i] = a[i] < b[i] ? a[i] : b[i] (signed comparison).
 *   - Boundary values injected at indices 0-3 (signed-vs-unsigned sign trap).
 *   - Output buffer poisoned to 0xA5.
 *   - HVX_TIME_KERNEL wraps the candidate call; HVXENV_KCYCLES is emitted.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#define N 500000

/* Static buffers aligned for HVX (128-byte) loads/stores. */
static int8_t a[N]   HVX_ALIGN;
static int8_t b[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    /* --- Fast HVX fill of a[] and b[] using deterministic byte patterns.
     * Pattern: a[i] = (int8_t)(i*7+3), b[i] = (int8_t)(i*11+5).
     * Implemented as HVX vector stores for speed; scalar tail for the remainder. */
    {
        const int vlen = sizeof(HVX_Vector);
        /* Build base vectors and per-step deltas. */
        int8_t va_init[128], vb_init[128], va_step[128], vb_step[128];
        for (int j = 0; j < 128; j++) {
            va_init[j] = (int8_t)(j * 7 + 3);
            vb_init[j] = (int8_t)(j * 11 + 5);
            va_step[j] = (int8_t)(128 * 7);   /* per-vector stride (mod 256) */
            vb_step[j] = (int8_t)(128 * 11);
        }
        HVX_Vector cur_a = *(HVX_Vector *)va_init;
        HVX_Vector cur_b = *(HVX_Vector *)vb_init;
        HVX_Vector step_a = *(HVX_Vector *)va_step;
        HVX_Vector step_b = *(HVX_Vector *)vb_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            *(HVX_Vector *)(a + i) = cur_a;
            *(HVX_Vector *)(b + i) = cur_b;
            cur_a = Q6_Vb_vadd_VbVb(cur_a, step_a);
            cur_b = Q6_Vb_vadd_VbVb(cur_b, step_b);
        }
        for (; i < N; i++) { a[i] = (int8_t)(i*7+3); b[i] = (int8_t)(i*11+5); }
    }

    /* Injected boundary values: signed-vs-unsigned min discriminators.
     * Unsigned interpretation would treat -1 (0xFF) and -128 (0x80) as LARGE,
     * flipping the min result versus the correct signed comparison. */
    a[0]=-1;    b[0]=1;    /* signed min(-1,1)=-1; unsigned min(0xFF,0x01)=0x01=1 (wrong) */
    a[1]=1;     b[1]=-1;   /* signed min(1,-1)=-1; unsigned would pick 1 (wrong) */
    a[2]=127;   b[2]=-128; /* signed min=-128; extreme range span */
    a[3]=-128;  b[3]=-128; /* tie: signed min(-128,-128)=-128 */

    /* --- Compute reference via HVX (same op as a correct candidate_kernel). */
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = Q6_Vb_vmin_VbVb(
                *(const HVX_Vector *)(a + i), *(const HVX_Vector *)(b + i));
        for (; i < N; i++) ref[i] = (a[i] < b[i]) ? a[i] : b[i];
    }

    /* --- Poison output buffer (HVX broadcast store of 0xA5). */
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; memset(poison_arr, 0xA5, 128);
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    /* --- Timed candidate call. */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* --- Verify: HVX bulk check first, scalar scan only on mismatch. */
    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0;
        int i = 0;
        for (; i + vlen <= N && !mismatch; i += vlen) {
            HVX_VectorPred diff = Q6_Q_vcmp_eq_VbVb(
                *(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            /* Q6_Q_vcmp_eq returns 1-bits where equal; any 0 means mismatch. */
            /* Simplest portable test: extract the predicate to check all bytes. */
            int8_t tmp[128] HVX_ALIGN;
            *(HVX_Vector *)tmp = Q6_V_vand_QR(diff, -1);  /* 0xFF where equal */
            for (int j = 0; j < vlen; j++) {
                if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
            }
        }
        for (; i < N; i++) {
            if (out[i] != ref[i]) { mismatch = 1; break; }
        }
        if (mismatch) {
            /* Slow path: full scalar scan to count and locate errors. */
            for (int i2 = 0; i2 < N; i2++) {
                if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
            }
        }
    }

    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
