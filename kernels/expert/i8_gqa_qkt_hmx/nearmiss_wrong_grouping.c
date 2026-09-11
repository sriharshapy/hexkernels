/* Near-miss: identical HMX tiled flow + weight-crouton reuse across the
 * shared KV head, but forgets to advance the query pointer per head (a
 * plausible copy-paste bug when generalizing a single-head kernel to GQA) --
 * every query head reads Q's HEAD-0 data. Compiles, uses real HMX, correct
 * for h_q==0, but wrong for h_q>=1 -> must FAIL the gate. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

#define T  32
#define NT (GQA_S / T)

void candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgtTiles[NT][NT];
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);
    for (int tj = 0; tj < NT; tj++)
        for (int kt = 0; kt < NT; kt++)
            vWgtTiles[tj][kt] = (int8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u + (unsigned)(tj*NT + kt) * 0x400u);

    static uint8_t  aAct[2048] HVX_ALIGN;
    static int8_t   aWgt[1024] HVX_ALIGN;
    static uint16_t aOut[T*T]  HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;

    const int8_t *K0 = K;
    for (int tj = 0; tj < NT; tj++) {
        for (int kt = 0; kt < NT; kt++) {
            for (int k = 0; k < T; k++)
                for (int j = 0; j < T; j++)
                    aWgt[hvx_hmx_i8_wgt_off(k, j)] = K0[(tj*T + j)*GQA_D + (kt*T + k)];
            { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgtTiles[tj][kt];
              for (int b = 0; b < 8; b++) d[b] = s[b]; }
        }
    }

    for (int hq = 0; hq < GQA_H_Q; hq++) {
        const uint8_t *Qh = Q;   /* BUG: should be Q + hq*GQA_S*GQA_D */
        uint16_t      *Oh = out + (size_t)hq * GQA_S * GQA_S;
        for (int ti = 0; ti < NT; ti++) {
            for (int tj = 0; tj < NT; tj++) {
                __asm__ volatile("mxclracc\n");
                for (int kt = 0; kt < NT; kt++) {
                    for (int i = 0; i < T; i++)
                        for (int k = 0; k < T; k++)
                            aAct[hvx_hmx_i8_act_off(i, k)] = Qh[(ti*T + i)*GQA_D + (kt*T + k)];
                    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
                      for (int b = 0; b < 16; b++) d[b] = s[b]; }
                    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                     :: "r"(vAct), "r"(lim_a), "r"(vWgtTiles[tj][kt]), "r"(lim_w) : "memory");
                }
                __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
                __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
                __asm__ volatile("isync\n\t");
                { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                for (int i = 0; i < T; i++)
                    for (int j = 0; j < T; j++)
                        Oh[(ti*T + i)*GQA_S + (tj*T + j)] = aOut[hvx_hmx_i8_out_off(i, j)];
            }
        }
    }
}
