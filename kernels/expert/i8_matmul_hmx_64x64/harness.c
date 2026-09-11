#include "harness_common.h"
#include "kernel_api.h"
#define N 64

static uint8_t  A[N*N] HVX_ALIGN;
static int8_t   B[N*N] HVX_ALIGN;
static uint16_t out[N*N] HVX_ALIGN;
static uint16_t ref[N*N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x64A1u;
    /* A in 0..7 (positive: int8==uint8), B in -3..3. With K=64 the worst-case
     * |acc| = 64*7*3 = 1344 -> |acc*17/16| = 1428 < 2048, so the 12-bit requant
     * field is exact (no HMX saturation) and bit-exactness holds. */
    for (int i = 0; i < N*N; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 8);          /* 0..7 */
    for (int i = 0; i < N*N; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3); /* -3..3 */

    /* scalar reference: row-major matmul + HMX 0x40-config requant (12-bit field) */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int acc = 0;
            for (int k = 0; k < N; k++) acc += (int)A[i*N+k] * (int)B[k*N+j];
            ref[i*N+j] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }

    for (int i = 0; i < N*N; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        unsigned short g = out[i], e = ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; } }
    }
    hvx_report_u16(errors, N*N, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
