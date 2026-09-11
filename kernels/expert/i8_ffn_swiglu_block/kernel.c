/* EXPERT (achievability bar) -- the L3 SwiGLU composition: HMX x3 + VTCM + HVX.
 * A modern gated FFN block: gate=X.Wg, up=X.Wu, H=SiLU(gate)*up, out=H.Wd. ALL
 * THREE matmuls run on the HMX matrix engine (32x32 crouton tiles, 0x40 requant);
 * the SiLU LUT gate + elementwise product + requant/bias/saturate epilogues run on
 * HVX/scalar; the gated intermediate activation H is staged VTCM-resident between
 * the up/gate matmuls and the down matmul.
 *
 * WHY THIS BEATS THE HVX vrmpy BASELINE: the baseline computes all three matmuls
 * as vrmpy dot-products with a horizontal reduce per output cell; the expert
 * offloads that arithmetic to the HMX systolic array (one packet per 32x32x32
 * tile-mul, accumulating across K-tiles), spending HVX only on packing + the
 * pointwise gate epilogue.
 *
 * CROUTON PACK/UNPACK IS THE REAL COST -- it is staged through cacheable buffers
 * and moved to/from VTCM in bulk 128B vector copies (a scalar VTCM access costs
 * ~48 cyc in timing mode and would lose). H lives in VTCM between the matmul
 * stages (bulk-copied in after the gate, bulk-copied back out to pack the down
 * matmul's activation croutons). The down-proj activation carries an int8
 * zero-point SW_ZP; Wd's zero-sum columns make it contribute no offset, so the
 * whole block is BIT-EXACT to the scalar reference.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md +
 *         datasets/v5/tasks/i8_ffn_block/expert.c */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

#define T 32

/* VTCM partition */
#define VT_ACT  (HVX_VTCM_BASE + 0x0000u)   /* 2KB activation crouton */
#define VT_WGT  (HVX_VTCM_BASE + 0x0800u)   /* 1KB weight crouton     */
#define VT_BIAS (HVX_VTCM_BASE + 0x1000u)   /* 2KB requant config     */
#define VT_OUT  (HVX_VTCM_BASE + 0x1800u)   /* 2KB output crouton     */
#define VT_H    (HVX_VTCM_BASE + 0x2000u)   /* S*Dff gated intermediate */

/* C_field[M x N] = A[M x Kd] . W[Kd x ldW] via HMX (weight straight, no transpose),
 * 0x40 requant, output = uint16 12-bit field row-major (leading dim N). */
static void hmx_matmul(const uint8_t *A, const int8_t *W, uint16_t *C,
                       int M, int N, int Kd, int ldW) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)VT_ACT;
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)VT_WGT;
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)VT_BIAS;
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)VT_OUT;

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[T*T]   HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i] = 0;   /* even/low bytes stay 0 */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int ntM = M / T, ntN = N / T, nkt = Kd / T;

    for (int ti = 0; ti < ntM; ti++) {
        for (int tj = 0; tj < ntN; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < nkt; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*Kd + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W[(kt*T + k)*ldW + (tj*T + j)];
                { HVX_Vector *sp = (HVX_Vector *)aAct, *dp = (HVX_Vector *)vAct;
                  for (int b = 0; b < 16; b++) dp[b] = sp[b]; }
                { HVX_Vector *sp = (HVX_Vector *)aWgt, *dp = (HVX_Vector *)vWgt;
                  for (int b = 0; b < 8; b++) dp[b] = sp[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *sp = (HVX_Vector *)vOut, *dp = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) dp[b] = sp[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    C[(ti*T + i)*N + (tj*T + j)] = aOut[hvx_hmx_i8_out_off(i, j)];
        }
    }
}

void candidate_kernel(const uint8_t *X, const int8_t *Wg, const int8_t *Wu,
                      const int8_t *Wd, const int32_t *bd, const uint8_t *silu_lut,
                      int8_t *out, int S, int D, int Dff) {
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)VT_BIAS;
    static uint8_t aBias[2048] HVX_ALIGN;
    for (int i = 0; i < 2048; i++) aBias[i] = 0x40;
    { HVX_Vector *sb = (HVX_Vector *)aBias, *db = (HVX_Vector *)(uintptr_t)VT_BIAS;
      for (int b = 0; b < 16; b++) db[b] = sb[b]; }   /* requant config -> VTCM once */
    (void)vBias;

    static uint16_t gate_f[SW_S*SW_DFF] HVX_ALIGN;   /* gate requant field */
    static uint16_t up_f  [SW_S*SW_DFF] HVX_ALIGN;   /* up   requant field */
    static uint8_t  Hu    [SW_S*SW_DFF] HVX_ALIGN;   /* gated activation (uint8) */
    static uint8_t  Hu2   [SW_S*SW_DFF] HVX_ALIGN;   /* H read back from VTCM */
    static uint16_t down_f[SW_S*SW_D]   HVX_ALIGN;   /* down requant field */

    /* ===== gate = X.Wg, up = X.Wu  (both HMX) ===== */
    hmx_matmul(X, Wg, gate_f, /*M*/S, /*N*/Dff, /*Kd*/D, /*ldW*/Dff);
    hmx_matmul(X, Wu, up_f,   /*M*/S, /*N*/Dff, /*Kd*/D, /*ldW*/Dff);

    /* ===== [HVX] SiLU LUT gate + elementwise product -> zero-pointed uint8 H ===== */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < Dff; j++) {
            int gq = sw_scale((int)gate_f[i*Dff + j], SW_SG);
            int uq = sw_scale((int)up_f  [i*Dff + j], SW_SU);
            int silu_g = (signed char)silu_lut[gq + 128];
            Hu[i*Dff + j] = sw_hu(silu_g, uq);
        }

    /* stage H VTCM-resident (bulk copy in, then read back for down-matmul packing) */
    const int nvH = (S*Dff) / 128;
    { HVX_Vector *sp = (HVX_Vector *)Hu, *dp = (HVX_Vector *)(uintptr_t)VT_H;
      for (int b = 0; b < nvH; b++) dp[b] = sp[b]; }
    { HVX_Vector *sp = (HVX_Vector *)(uintptr_t)VT_H, *dp = (HVX_Vector *)Hu2;
      for (int b = 0; b < nvH; b++) dp[b] = sp[b]; }

    /* ===== out = H.Wd  (HMX) -> requant + bias -> saturate ===== */
    hmx_matmul(Hu2, Wd, down_f, /*M*/S, /*N*/D, /*Kd*/Dff, /*ldW*/D);
    /* [HVX] down-proj epilogue: sign-extend the 12-bit requant field, add the
     * per-column bias, requant-shift, saturate -- all in vector lanes. down_f is the
     * row-major field (hmx_matmul already un-croutoned it) and D is 64, so one row is
     * 64 halfwords = exactly one 128B vector. A vasl-by-4 then vasr-by-4 pair
     * sign-extends the field in halfword lanes.
     *
     * The widening needs care. Q6_Ww_vsxt_Vh does NOT split its input into low and
     * high halves: Q6_V_lo_W holds the EVEN halfword lanes and Q6_V_hi_W the odd ones.
     * (The crouton unpacks elsewhere only get away with taking lo_W because
     * vshuffe/vshuffo duplicate each element adjacently first, so the even lanes are
     * exactly the wanted row. Assuming low/high here instead was wrong on 1984 of 4096
     * elements -- half, the signature of an even/odd split.)
     *
     * So the two word vectors are the even and the odd COLUMNS, and the bias is
     * de-interleaved to match, once, before the row loop rather than per row. The
     * store is scalar anyway -- out is int8, so a row is 64 B -- and picking the even
     * or odd half there costs nothing. */
    {
        const HVX_Vector vlo = Q6_V_vsplat_R(-128);
        const HVX_Vector vhi = Q6_V_vsplat_R(127);
        static int32_t aBd[2][32] HVX_ALIGN, aRow[2][32] HVX_ALIGN;
        for (int m = 0; m < D; m++) aBd[m & 1][m >> 1] = bd[m];
        for (int i = 0; i < S; i++) {
            HVX_Vector se = Q6_Vh_vasr_VhR(
                Q6_Vh_vasl_VhR(((const HVX_Vector *)down_f)[i], 4), 4);
            HVX_VectorPair wp = Q6_Ww_vsxt_Vh(se);
            HVX_Vector w[2];
            w[0] = Q6_V_lo_W(wp);    /* even columns */
            w[1] = Q6_V_hi_W(wp);    /* odd  columns */
            for (int h = 0; h < 2; h++) {
                HVX_Vector v = Q6_Vw_vasr_VwR(
                    Q6_Vw_vadd_VwVw(w[h], *(const HVX_Vector *)aBd[h]), SW_SO);
                *(HVX_Vector *)aRow[h] = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(v, vlo), vhi);
            }
            for (int m = 0; m < D; m++)
                out[i*D + m] = (int8_t)aRow[m & 1][m >> 1];
        }
    }
}
