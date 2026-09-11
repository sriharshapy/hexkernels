#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#include <string.h>

#define BLK 24   /* NOT a multiple of 64 fp16 lanes */
#define D   48   /* NOT a multiple of 64 fp16 lanes */

static hvx_hf scores[BLK] HVX_ALIGN;
static hvx_hf V[BLK*D]    HVX_ALIGN;
static hvx_hf acc[D]      HVX_ALIGN;
static hvx_hf rmax, rsum;

typedef struct { float m; float l; float acc[D]; } State;

static float gen_hf(uint32_t *s, int range_q, float step) {
    int q = (int)(hvx_lcg(s) % (unsigned)(2*range_q+1)) - range_q;
    return (float)(q * step);
}

static unsigned short hf_bits(float f) {
    hvx_hf h = (hvx_hf)f;
    unsigned short b;
    memcpy(&b, &h, 2);
    return b;
}

/* Independent scalar float reference: same recurrence as kernel_api.h. */
static void ref_step(const hvx_hf *sc, const hvx_hf *v, State *st) {
    float m_old = st->m, l_old = st->l;

    float m_blk = (float)sc[0];
    for (int j = 1; j < BLK; j++) { float x = (float)sc[j]; if (x > m_blk) m_blk = x; }
    float m_new = (m_old > m_blk) ? m_old : m_blk;
    float corr  = expf(m_old - m_new);

    float p[BLK]; float psum = 0.0f;
    for (int j = 0; j < BLK; j++) { p[j] = expf((float)sc[j] - m_new); psum += p[j]; }
    float l_new = l_old * corr + psum;

    for (int d = 0; d < D; d++) {
        float a = st->acc[d] * corr;
        for (int j = 0; j < BLK; j++) a += p[j] * (float)v[j*D + d];
        st->acc[d] = a;
    }
    st->m = m_new;
    st->l = l_new;
}

static int errors = 0, fb = -1, total = 0;
static long gotv = 0, expv = 0;

static void check_u16(unsigned short g, unsigned short e, int idx) {
    total++;
    if (!hvx_close_f16bits(g, e)) {
        errors++;
        if (fb < 0) { fb = idx; gotv = (long)g; expv = (long)e; }
    }
}

/* Run one test case: given initial state + inputs, call the candidate and
 * compare its updated (running_max, running_sum, acc[]) against the
 * independent reference. */
static void run_case(int case_id, float init_m, float init_l, float init_acc,
                     uint32_t seed, int score_bias_idx, float score_bias_val) {
    uint32_t s = seed;
    for (int i = 0; i < BLK; i++) scores[i] = (hvx_hf)gen_hf(&s, 8, 0.5f);
    if (score_bias_idx >= 0) scores[score_bias_idx] = (hvx_hf)score_bias_val;
    for (int i = 0; i < BLK*D; i++) V[i] = (hvx_hf)gen_hf(&s, 4, 0.5f);

    rmax = (hvx_hf)init_m;
    rsum = (hvx_hf)init_l;
    for (int d = 0; d < D; d++) acc[d] = (hvx_hf)init_acc;

    State st;
    st.m = init_m; st.l = init_l;
    for (int d = 0; d < D; d++) st.acc[d] = init_acc;
    ref_step(scores, V, &st);

    /* NOTE: acc[] is an in/out running state (not a pure output), so we do
     * NOT poison it -- its pre-call value is the semantically meaningful
     * prior partial-softmax accumulator, already set above. */
    unsigned long long kc = 0;
    HVX_TIME_KERNEL(kc, { candidate_kernel(scores, V, &rmax, &rsum, acc, BLK, D); });
    printf("HVXENV_KCYCLES kernel=%llu\n", kc);

    check_u16(*(unsigned short *)&rmax, hf_bits(st.m), case_id*1000 + 0);
    check_u16(*(unsigned short *)&rsum, hf_bits(st.l), case_id*1000 + 1);
    for (int d = 0; d < D; d++)
        check_u16(*(unsigned short *)&acc[d], hf_bits(st.acc[d]), case_id*1000 + 2 + d);
}

int main(void) {
    /* Case 0: first block -- default init state (running_max very negative
     * but finite, running_sum=0, acc=0). corr==1 trivially (m_new==m_old
     * only if m_blk<=m_old, but m_old=-30000 so m_new=m_blk always here;
     * corr = exp(m_old - m_new) = exp(-30000 - m_blk) ~= 0, exercising the
     * "old state contributes ~nothing" branch). */
    run_case(0, -30000.0f, 0.0f, 0.0f, 0xA5C31u, -1, 0.0f);

    /* Case 1: second block with a REAL rescale -- prior state has a
     * moderate running_max/running_sum/acc, and the new block's max score
     * is forced above the prior running_max (m_new > m_old, corr < 1). */
    run_case(1, 1.0f, 3.0f, 0.5f, 0xBEEF77u, 3, 5.0f);

    /* Case 2: second block where the block max does NOT exceed the prior
     * running_max (m_new == m_old, corr == 1 exactly) but the prior state
     * is nonzero -- distinguishes "corr==1 with nonzero prior" from case 0's
     * "corr==1 with all-zero prior". */
    run_case(2, 20.0f, 2.0f, 1.5f, 0xC0DE99u, -1, 0.0f);

    hvx_report_u16(errors, total, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
