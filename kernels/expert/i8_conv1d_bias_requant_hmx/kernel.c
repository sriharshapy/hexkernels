/* HMX int8 DENSE 1D conv (VALID) + bias + int8 requant via im2col — ACCELERATED expert.
 * RE-ROUTE of the compute-bound i8_conv1d_bias_requant_dma FAIL: expressed as the
 * matmul-lowerable dense cross-channel conv1d and run on the HMX matrix engine.
 * Lower to a matmul: rows = output positions ol (M=OL), reduction = (ci,kw) receptive
 * field (K = C_in*Kw = 80), cols = output channels (N=C_out). im2col gathers each
 * output position's window into a crouton row; K=80 is zero-padded to 3 crouton K-tiles.
 *
 * KEY: the im2col activation for an M-tile is packed into VTCM ONCE and REUSED across all
 * output-channel (N) tiles — the natural wide-output NPU pattern that lets the HMX matmul
 * throughput amortize the im2col cost (repacking per N-tile loses). Division-free pack via
 * precomputed koff (= ci*L_in + kw) + poff (= ol); weight value = W[co*KK + kg]. HVX unpack
 * fuses sign-extend + per-channel int32 bias + int8 saturating requant. Crouton pack/unpack
 * staged via cacheable buffers + bulk 128B vector copies.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      int8_t *out, int n) {
    const int T = 32;
    const int mt = (P_OUT + T - 1) / T;
    const int nt = (C_OUT + T - 1) / T;
    const int kn = (KK   + T - 1) / T;

    /* VTCM: kn activation tiles resident (2048B each), one weight tile, bias, output */
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u); /* kn*2048 */
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u); /* 1024 */
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3800u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x4800u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 1024; i++) aWgt[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    int koff[KK], poff[P_OUT];
    for (int k = 0; k < KK; k++) { int ci = k / KW, kw = k % KW; koff[k] = ci*L_IN + kw; }
    for (int p = 0; p < P_OUT; p++) poff[p] = p;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;

    for (int ti = 0; ti < mt; ti++) {
        /* pack im2col activation for this M-tile ONCE (all K-tiles) into VTCM */
        for (int kt = 0; kt < kn; kt++) {
            for (int i = 0; i < 2048; i++) aAct[i] = 0;
            for (int i = 0; i < T; i++) {
                int p = ti*T + i;
                if (p >= P_OUT) continue;
                int po = poff[p];
                for (int k = 0; k < T; k++) {
                    int kg = kt*T + k;
                    aAct[hvx_hmx_i8_act_off(i, k)] = (kg < KK) ? in[koff[kg] + po] : 0;
                }
            }
            HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)(vAct + kt*2048);
            for (int b = 0; b < 16; b++) d[b] = s[b];
        }
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < kn; kt++) {
                for (int k = 0; k < T; k++) {
                    int kg = kt*T + k;
                    for (int j = 0; j < T; j++) {
                        int co = tj*T + j;
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] =
                            (co < C_OUT && kg < KK) ? W[co*KK + kg] : 0;
                    }
                }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                uint8_t *actk = vAct + kt*2048;
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(actk), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            /* Fused HVX epilogue. The crouton output layout is off(r,c) = (r/2)*64 +
             * c*2 + (r&1), so ONE 128B vector is exactly one row PAIR: 32 columns of
             * the even row interleaved with 32 of the odd. vshuffe/vshuffo split it
             * into the two rows -- both land in the LOW 32 lanes, which is why only
             * Q6_V_lo_W of the widening pair is taken (a vdeal + lo/hi version failed
             * exactly half the elements, the signature of a wrong upper-half ordering
             * assumption). vasl/vasr sign-extends the 12-bit requant field, vsxt widens
             * to int32, and the per-channel bias is a word-lane vadd -- the tile COLUMN
             * j is the output channel, so bias[tj*T + j] is indexed by lane and loads as
             * one aligned vector (C_OUT is a multiple of 32). The clamp to [-128,127] is
             * left to the word lanes with vmin/vmax rather than a vpack:sat chain,
             * because the store transposes anyway and never packs to bytes.
             *
             * The STORE cannot be a vector store: out is [C_OUT][OL], so a tile row
             * scatters down an output column with stride OL. The 32 results go to an
             * aligned scratch in one vector store and are handed off from there. */
            {
                const HVX_Vector vb   = *(const HVX_Vector *)(bias + tj*T);
                const HVX_Vector vlo  = Q6_V_vsplat_R(-128);
                const HVX_Vector vhi  = Q6_V_vsplat_R(127);
                static int32_t aRow[32] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb);
                    w[1] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb);
                    for (int h = 0; h < 2; h++) {
                        int p = ti*T + 2*rp + h;
                        if (p >= P_OUT) continue;
                        *(HVX_Vector *)aRow =
                            Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(w[h], vlo), vhi);
                        for (int j = 0; j < T; j++) {
                            int co = tj*T + j;
                            if (co >= C_OUT) continue;
                            out[co*OL + p] = (int8_t)aRow[j];
                        }
                    }
                }
            }
        }
    }
    (void)n;
}
