/* D=128 (the head-dim reduction axis) is a much deeper fp16/qf16 reduction
 * than the shallow int8 QK^T sibling was tuned for -- accumulating 128 terms
 * in 11-bit-mantissa qf16 drifts more than an IEEE float32 scalar reference,
 * especially near cancellation. Widen the fp16 tolerance accordingly (still
 * tight: ~1% relative). */
#define HVX_FP16_ATOL 8e-3f
#define HVX_FP16_RTOL 1.2e-2f
#include "harness_common.h"
#include "kernel_api.h"
#define N   32   /* S: single output crouton tile */
#define KD 128   /* D: deep head-dim reduction */

static hvx_hf Q[N*KD] HVX_ALIGN, K[N*KD] HVX_ALIGN;
static hvx_hf out[N*N] HVX_ALIGN, ref[N*N] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 7) - 3;   /* -3..3 */
    return (float)(q * 0.25f);           /* exact in fp16, |acc| bounded */
}

int main(void) {
    uint32_t s = 0x9C412u;
    for (int i = 0; i < N*KD; i++) Q[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < N*KD; i++) K[i] = (hvx_hf)gen_hf(&s);

    /* scalar reference: scores = Q . K^T (K row j is key vector j), float
     * accumulate over the full D=128 reduction, cast to fp16 */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int d = 0; d < KD; d++) acc += (float)Q[i*KD+d] * (float)K[j*KD+d];
            ref[i*N+j] = (hvx_hf)acc;
        }

    for (int i = 0; i < N*N; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, out, N, KD); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* fp16 tolerance compare (HVX/HMX qf16 arithmetic is non-IEEE) */
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        unsigned short g = *(unsigned short *)&out[i], e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, N*N, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
