/* K=128 (Skv, the reduction axis over keys/values) is a much deeper
 * fp16/qf16 reduction than the shallow int8 AV sibling was tuned for --
 * accumulating 128 terms in 11-bit-mantissa qf16 drifts more than an IEEE
 * float32 scalar reference, especially near cancellation. Widen the fp16
 * tolerance accordingly (still tight: ~1% relative). */
#define HVX_FP16_ATOL 8e-3f
#define HVX_FP16_RTOL 1.2e-2f
#include "harness_common.h"
#include "kernel_api.h"
#define N   32   /* S == D == 32 (single output crouton tile) */
#define KV 128   /* Skv: deep reduction over 128 keys/values */

static hvx_hf P[N*KV] HVX_ALIGN, V[KV*N] HVX_ALIGN;
static hvx_hf out[N*N] HVX_ALIGN, ref[N*N] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 7) - 3;   /* -3..3 */
    return (float)(q * 0.25f);           /* exact in fp16, |acc| bounded */
}

int main(void) {
    uint32_t s = 0x5A17Fu;
    for (int i = 0; i < N*KV; i++) P[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < KV*N; i++) V[i] = (hvx_hf)gen_hf(&s);

    /* scalar reference: float accumulate over the full Skv=128 reduction, cast to fp16 */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < KV; k++) acc += (float)P[i*KV+k] * (float)V[k*N+j];
            ref[i*N+j] = (hvx_hf)acc;
        }

    for (int i = 0; i < N*N; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(P, V, out, N, KV); });
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
