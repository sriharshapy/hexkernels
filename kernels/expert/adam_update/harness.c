#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N    256
#define LR   0.001f
#define B1   0.9f
#define B2   0.999f
#define EPS  1e-8f
#define T    10

/* Input arrays (originals, read-only after init) */
static float w_orig[N]    HVX_ALIGN;
static float m_orig[N]    HVX_ALIGN;
static float v_orig[N]    HVX_ALIGN;
static float grad[N]      HVX_ALIGN;

/* Work arrays passed to candidate (mutated in-place) */
static float w_work[N]    HVX_ALIGN;
static float m_work[N]    HVX_ALIGN;
static float v_work[N]    HVX_ALIGN;

/* Reference arrays (scalar computation on originals) */
static float w_ref[N]     HVX_ALIGN;
static float m_ref[N]     HVX_ALIGN;
static float v_ref[N]     HVX_ALIGN;

/* Generate fp32 in [-0.5, 0.5] with 0.0625 step. */
static float gen_f32_half(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 17) - 8;  /* -8..8 */
    return (float)q * 0.0625f;
}

/* Generate fp32 in [0, 0.25] for v_orig (must be non-negative for second moment). */
static float gen_f32_pos(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 9);  /* 0..8 */
    return (float)q * 0.03125f;
}

int main(void) {
    uint32_t s = 0xDEADBEEFu;
    for (int i = 0; i < N; i++) w_orig[i] = gen_f32_half(&s);
    for (int i = 0; i < N; i++) m_orig[i] = gen_f32_half(&s);
    for (int i = 0; i < N; i++) v_orig[i] = gen_f32_pos(&s);
    for (int i = 0; i < N; i++) grad[i]   = gen_f32_half(&s);

    /* ---- Pinned edge cases ---- */
    /* Edge 1: grad=0 -> m and v decay, w changes only via decayed moments */
    w_orig[0]  = 0.5f;  m_orig[0] = 0.1f;  v_orig[0] = 0.01f;  grad[0] = 0.0f;
    /* Edge 2: grad very small -> eps dominates denominator */
    w_orig[1]  = 0.1f;  m_orig[1] = 0.0f;  v_orig[1] = 0.0f;   grad[1] = 1e-7f;
    /* Edge 3: large gradient */
    w_orig[2]  = 0.0f;  m_orig[2] = 0.0f;  v_orig[2] = 0.0f;   grad[2] = 0.5f;
    /* Edge 4: negative gradient -> w increases */
    w_orig[3]  = 0.0f;  m_orig[3] = 0.0f;  v_orig[3] = 0.0f;   grad[3] = -0.5f;
    /* Edge 5: tail path -- last element */
    w_orig[N-1] = 0.2f; m_orig[N-1] = 0.05f; v_orig[N-1] = 0.002f; grad[N-1] = 0.1f;

    /* Pre-compute bias correction constants (same for all elements at step t). */
    float bc1 = 1.0f - powf(B1, (float)T);  /* 1 - b1^t */
    float bc2 = 1.0f - powf(B2, (float)T);  /* 1 - b2^t */

    /* Scalar reference: computed from originals */
    for (int i = 0; i < N; i++) {
        float mi   = B1 * m_orig[i] + (1.0f - B1) * grad[i];
        float vi   = B2 * v_orig[i] + (1.0f - B2) * grad[i] * grad[i];
        float mhat = mi / bc1;
        float vhat = vi / bc2;
        float wi   = w_orig[i] - LR * mhat / (sqrtf(vhat) + EPS);
        m_ref[i] = mi;
        v_ref[i] = vi;
        w_ref[i] = wi;
    }

    /* Copy originals into work arrays for the candidate */
    for (int i = 0; i < N; i++) {
        w_work[i] = w_orig[i];
        m_work[i] = m_orig[i];
        v_work[i] = v_orig[i];
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(w_work, m_work, v_work, grad, N, LR, B1, B2, EPS, T); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* Compare all three updated arrays; n = 3*N total elements compared */
    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int i = 0; i < N; i++) {
        if (!hvx_close_f32(m_work[i], m_ref[i])) {
            errors++;
            if (fb < 0) {
                fb = i;
                __builtin_memcpy(&gotv, &m_work[i], 4);
                __builtin_memcpy(&expv, &m_ref[i],  4);
            }
        }
    }
    for (int i = 0; i < N; i++) {
        if (!hvx_close_f32(v_work[i], v_ref[i])) {
            errors++;
            if (fb < 0) {
                fb = N + i;
                __builtin_memcpy(&gotv, &v_work[i], 4);
                __builtin_memcpy(&expv, &v_ref[i],  4);
            }
        }
    }
    for (int i = 0; i < N; i++) {
        if (!hvx_close_f32(w_work[i], w_ref[i])) {
            errors++;
            if (fb < 0) {
                fb = 2 * N + i;
                __builtin_memcpy(&gotv, &w_work[i], 4);
                __builtin_memcpy(&expv, &w_ref[i],  4);
            }
        }
    }

    hvx_report(errors, 3 * N, fb, gotv, expv);
    return errors ? 1 : 0;
}
