#include "harness_common.h"
#include "kernel_api.h"
#define N   512
#define LR  0.01f
#define MU  0.9f

/* Input arrays (originals, read-only after init) */
static float w_orig[N]    HVX_ALIGN;
static float v_orig[N]    HVX_ALIGN;
static float grad[N]      HVX_ALIGN;

/* Work arrays passed to candidate (mutated in-place) */
static float w_work[N]    HVX_ALIGN;
static float v_work[N]    HVX_ALIGN;

/* Reference arrays (scalar computation on originals) */
static float w_ref[N]     HVX_ALIGN;
static float v_ref[N]     HVX_ALIGN;

/* Generate fp32 in [-1, 1] with 0.125 step. */
static float gen_f32_1(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 17) - 8;  /* -8..8 -> -1.0..1.0 */
    return (float)q * 0.125f;
}

int main(void) {
    uint32_t s = 0xDEADBEEFu;
    for (int i = 0; i < N; i++) w_orig[i]  = gen_f32_1(&s);
    for (int i = 0; i < N; i++) v_orig[i]  = gen_f32_1(&s);
    for (int i = 0; i < N; i++) grad[i]    = gen_f32_1(&s);

    /* ---- Pinned edge cases ---- */
    /* Edge 1: grad=0, v non-zero -> v decays by mu, w updated by decayed v */
    w_orig[0]  = 0.5f;   v_orig[0] = 0.8f;   grad[0] = 0.0f;
    /* Edge 2: large positive gradient */
    w_orig[1]  = 0.0f;   v_orig[1] = 0.0f;   grad[1] = 1.0f;
    /* Edge 3: negative gradient -> w increases */
    w_orig[2]  = 0.0f;   v_orig[2] = 0.0f;   grad[2] = -1.0f;
    /* Edge 4: momentum carry-over */
    w_orig[3]  = 0.25f;  v_orig[3] = 0.5f;   grad[3] = 0.25f;
    /* Edge 5: tail path -- last element */
    w_orig[N-1] = 0.1f; v_orig[N-1] = 0.2f; grad[N-1] = 0.3f;

    /* Scalar reference: computed from originals, stored in ref arrays */
    for (int i = 0; i < N; i++) {
        float vi = MU * v_orig[i] + grad[i];
        float wi = w_orig[i] - LR * vi;
        v_ref[i] = vi;
        w_ref[i] = wi;
    }

    /* Copy originals into work arrays for the candidate */
    for (int i = 0; i < N; i++) { w_work[i] = w_orig[i]; v_work[i] = v_orig[i]; }

    /* Poison check: work arrays are already initialized; candidate updates them in place */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(w_work, v_work, grad, N, LR, MU); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* Compare both updated arrays; n = 2*N total elements compared */
    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int i = 0; i < N; i++) {
        if (!hvx_close_f32(v_work[i], v_ref[i])) {
            errors++;
            if (fb < 0) {
                fb = i;  /* index into v array */
                __builtin_memcpy(&gotv, &v_work[i], 4);
                __builtin_memcpy(&expv, &v_ref[i],  4);
            }
        }
    }
    for (int i = 0; i < N; i++) {
        if (!hvx_close_f32(w_work[i], w_ref[i])) {
            errors++;
            if (fb < 0) {
                fb = N + i;  /* offset to distinguish w errors from v errors */
                __builtin_memcpy(&gotv, &w_work[i], 4);
                __builtin_memcpy(&expv, &w_ref[i],  4);
            }
        }
    }

    hvx_report(errors, 2 * N, fb, gotv, expv);
    return errors ? 1 : 0;
}
