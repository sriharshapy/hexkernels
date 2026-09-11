#include "harness_common.h"
#include "kernel_api.h"
#define N 32

static uint8_t A[N*N] HVX_ALIGN;
static int8_t  B[N*N] HVX_ALIGN;
static int8_t  out[N*N] HVX_ALIGN;
static int8_t  ref[N*N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x9A1Cu;
    /* A in 0..3 (uint8), B in -1..1 (int8). Worst-case |acc| = 32*3*1 = 96, so
     * |acc*17/16| = 102 < 128 -- the requant-to-int8 cast never saturates and
     * stays bit-exact for every input in this domain. */
    for (int i = 0; i < N*N; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 4);           /* 0..3 */
    for (int i = 0; i < N*N; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 3) - 1); /* -1..1 */

    /* scalar reference: row-major matmul + HMX 0x40-config requant, narrowed to int8 */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int acc = 0;
            for (int k = 0; k < N; k++) acc += (int)A[i*N+k] * (int)B[k*N+j];
            int r = (acc * 17 + 8) >> 4;
            ref[i*N+j] = (int8_t)r;      /* exact: |r| <= 102 by input-range construction */
        }

    for (int i = 0; i < N*N; i++) *((volatile signed char *)&out[i]) = (signed char)0x55; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, N*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
