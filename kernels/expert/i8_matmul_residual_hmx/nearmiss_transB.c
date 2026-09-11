/* Near-miss: identical HMX tiled flow + residual-add epilogue, but packs B
 * TRANSPOSED (B[j*n+k] instead of B[k*n+j]) inside each K-tile, so it computes
 * (A*B^T requant) + C. Compiles, uses real HMX, still applies the residual add
 * -- but the matmul itself is numerically wrong -> must FAIL the bit-exact gate. */
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *C,
                       int32_t *out, int n) {
    volatile uint8_t  *vAct  = (volatile uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    volatile int8_t   *vWgt  = (volatile int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t  *vBias = (volatile uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    volatile uint16_t *vOut  = (volatile uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    for (int i = 0; i < 2048; i++) vAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int nt = n / T;

    for (int ti = 0; ti < nt; ti++) {
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < nt; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        vAct[hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*n + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        vWgt[hvx_hmx_i8_wgt_off(k, j)] = B[(tj*T + j)*n + (kt*T + k)]; /* transposed */
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++) {
                    int r  = sx12((int)vOut[hvx_hmx_i8_out_off(i, j)]);
                    int gi = ti*T + i, gj = tj*T + j;
                    out[gi*n + gj] = r + C[gi*n + gj];
                }
        }
    }
}
