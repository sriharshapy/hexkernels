/* i8_matvec sol_03: HVX with Q6_Vw_vrmpy_VbVb, 4-row batched, correct hsum via aligned store. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline int32_t hsum_vw(HVX_Vector v) {
    int32_t tmp[32] __attribute__((aligned(128)));
    *(HVX_Vector *)tmp = v;
    int32_t s = 0;
    for (int l = 0; l < 32; l++) s += tmp[l];
    return s;
}

static void pack128(int8_t *dst, const int8_t *src, int len) {
    for (int i = 0; i < len && i < 128; i++) dst[i] = src[i];
    for (int i = len; i < 128; i++) dst[i] = 0;
}

void candidate_kernel(const int8_t *A, const int8_t *x, int32_t *y,
                      int M, int K) {
    int8_t xbuf[128] __attribute__((aligned(128)));
    pack128(xbuf, x, K < 128 ? K : 128);
    HVX_Vector vx = *(const HVX_Vector *)xbuf;

    int8_t abuf0[128] __attribute__((aligned(128)));
    int8_t abuf1[128] __attribute__((aligned(128)));
    int8_t abuf2[128] __attribute__((aligned(128)));
    int8_t abuf3[128] __attribute__((aligned(128)));

    int i = 0;
    for (; i + 3 < M; i += 4) {
        pack128(abuf0, A + (i+0)*K, K < 128 ? K : 128);
        pack128(abuf1, A + (i+1)*K, K < 128 ? K : 128);
        pack128(abuf2, A + (i+2)*K, K < 128 ? K : 128);
        pack128(abuf3, A + (i+3)*K, K < 128 ? K : 128);
        int32_t acc0 = hsum_vw(Q6_Vw_vrmpy_VbVb(*(HVX_Vector *)abuf0, vx));
        int32_t acc1 = hsum_vw(Q6_Vw_vrmpy_VbVb(*(HVX_Vector *)abuf1, vx));
        int32_t acc2 = hsum_vw(Q6_Vw_vrmpy_VbVb(*(HVX_Vector *)abuf2, vx));
        int32_t acc3 = hsum_vw(Q6_Vw_vrmpy_VbVb(*(HVX_Vector *)abuf3, vx));
        for (int k = 128; k < K; k++) {
            acc0 += (int32_t)A[(i+0)*K+k]*(int32_t)x[k];
            acc1 += (int32_t)A[(i+1)*K+k]*(int32_t)x[k];
            acc2 += (int32_t)A[(i+2)*K+k]*(int32_t)x[k];
            acc3 += (int32_t)A[(i+3)*K+k]*(int32_t)x[k];
        }
        y[i+0]=acc0; y[i+1]=acc1; y[i+2]=acc2; y[i+3]=acc3;
    }
    for (; i < M; i++) {
        pack128(abuf0, A + i*K, K < 128 ? K : 128);
        int32_t acc = hsum_vw(Q6_Vw_vrmpy_VbVb(*(HVX_Vector *)abuf0, vx));
        for (int k = 128; k < K; k++) acc += (int32_t)A[i*K+k]*(int32_t)x[k];
        y[i] = acc;
    }
}