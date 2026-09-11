#include "harness_common.h"
#include "kernel_api.h"
#define M 20
#define N 20
#define K 80
static int8_t A[M*K] HVX_ALIGN, B[N*K] HVX_ALIGN;
static int32_t bias[N] HVX_ALIGN;
static int8_t C[M*N] HVX_ALIGN, ref[M*N] HVX_ALIGN;

int main(void){
    uint32_t s = 0x51A3B7u;
    for (int i=0;i<M*K;i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);            /* full signed range */
    for (int i=0;i<N*K;i++) B[i] = (int8_t)((hvx_lcg(&s) >> 24) & 0x7F);   /* restricted [0,127] (non-negative) */
    for (int j=0;j<N;j++) bias[j] = (int32_t)((int32_t)(hvx_lcg(&s) % 121) - 60); /* [-60,60] */

    /* Row0 of A = -1, row0 of B (=column0) = 1: forces acc(0,0) = -K = -80 exactly.
     * bias[0]=90 pushes acc(0,0)+bias = 10 -- POSITIVE -- verifying bias is added
     * BEFORE relu (a relu-first bug would give max(-80,0)+90=90 instead of 10). */
    for (int k=0;k<K;k++) { A[0*K+k] = -1; B[0*K+k] = 1; }
    bias[0] = 90;

    /* Row1 of A = 127, row1 of B (=column1) = 127: acc(1,1) = 127*127*K = 1,290,320,
     * hugely positive -- verifies the upper saturate clamp fires AFTER relu+bias. */
    for (int k=0;k<K;k++) { A[1*K+k] = 127; B[1*K+k] = 127; }

    /* Row (M-1) of A = -128, combined with B globally non-negative: acc(M-1,j) <= 0
     * for EVERY j (and in practice hugely negative), so the whole row must come out
     * all-zero after relu regardless of bias -- also guards against a row/column
     * broadcast-axis swap on the bias. */
    for (int k=0;k<K;k++) A[(M-1)*K+k] = -128;

    /* Row (M-2) of A = 0: acc(M-2,j) = 0 for every j, so output must equal
     * clamp(max(bias[j],0),0,127) EXACTLY -- verifies bias isn't accidentally
     * scaled/duplicated when fused into the epilogue. */
    for (int k=0;k<K;k++) A[(M-2)*K+k] = 0;

    /* An explicit B==0 extreme, independent of the other overrides. */
    B[(N-1)*K + 0] = 0;

    for (int i=0;i<M;i++) for (int j=0;j<N;j++) {
        int32_t acc=0;
        for (int k=0;k<K;k++) acc += (int32_t)A[i*K+k]*(int32_t)B[j*K+k];
        acc += bias[j];
        int32_t relu = acc > 0 ? acc : 0;
        int32_t v = relu > 127 ? 127 : relu;
        ref[i*N+j] = (int8_t)v;
    }
    for (int i=0;i<M*N;i++) C[i] = (int8_t)0xA5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A,B,bias,C,M,N,K); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors=0, fb=-1;
    for (int i=0;i<M*N;i++) if (C[i]!=ref[i]) { errors++; if(fb<0) fb=i; }
    hvx_report(errors, M*N, fb, fb>=0?(long)C[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
