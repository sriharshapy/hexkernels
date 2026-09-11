#include "harness_common.h"
#include "kernel_api.h"
#define M 24
#define N 24
#define K 100
static int8_t A[M*K] HVX_ALIGN, B[N*K] HVX_ALIGN;
static int32_t C[M*N] HVX_ALIGN, ref[M*N] HVX_ALIGN;
int main(void){
    uint32_t s = 0x9911EEu;
    for (int i=0;i<M*K;i++) A[i]=(int8_t)(hvx_lcg(&s)>>24);
    for (int i=0;i<N*K;i++) B[i]=(int8_t)(hvx_lcg(&s)>>24);
    A[0]=127; B[0]=127;                 /* large positive contribution (row0 vs row0) */
    A[1]=-128; B[K]=127;                /* sign in the K reduction (A row0 k=1, B row1 k=0) */
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        int32_t acc=0;
        for (int k=0;k<K;k++) acc += (int32_t)A[i*K+k]*(int32_t)B[j*K+k];
        ref[i*N+j]=acc;
    }
    for (int i=0;i<M*N;i++) C[i]=(int32_t)0xA5A5A5A5;   /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A,B,C,M,N,K); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for (int i=0;i<M*N;i++) if (C[i]!=ref[i]) { errors++; if(fb<0) fb=i; }
    hvx_report(errors, M*N, fb, fb>=0?(long)C[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
