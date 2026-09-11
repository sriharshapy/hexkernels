#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 32
#define KDIM 32

static hvx_hf A[N*KDIM] HVX_ALIGN, B[KDIM*N] HVX_ALIGN;
static hvx_hf out[N*N] HVX_ALIGN, ref[N*N] HVX_ALIGN;

static float sigmoid_f32(float x) { return 1.0f / (1.0f + expf(-x)); }

static float gen_hf(uint32_t *s) {
    /* Wider range than the add/relu siblings so sigmoid sees both saturating
     * and mid-range inputs. */
    int q = (int)(hvx_lcg(s) % 13) - 6;   /* -6..6 */
    return (float)(q * 0.25f);             /* exact in fp16 */
}

int main(void) {
    uint32_t s = 0xF16Cu;
    for (int i = 0; i < N*KDIM; i++) A[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < KDIM*N; i++) B[i] = (hvx_hf)gen_hf(&s);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < KDIM; k++) acc += (float)A[i*KDIM+k] * (float)B[k*N+j];
            hvx_hf m = (hvx_hf)acc;          /* fp16-round */
            ref[i*N+j] = (hvx_hf)sigmoid_f32((float)m);
        }

    for (int i = 0; i < N*N; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, N, KDIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

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
