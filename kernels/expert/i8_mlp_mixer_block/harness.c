/* i8_mlp_mixer_block harness (L3 composite, holdout group H).
 * Token-mixing MLP + residual, then channel-mixing MLP + residual, all int8
 * fixed-point. The harness owns main(): seeds deterministic inputs, computes
 * the FULL composed scalar reference (plain integer loops), poisons the
 * output, times the candidate (kernel-only pcycles), and does a bit-exact
 * compare. */
#include "harness_common.h"
#include "kernel_api.h"

#define S   32
#define D   64
#define ST  64
#define DFF 64

static uint8_t X  [S*D]    HVX_ALIGN;
static int8_t  Wt1[ST*S]   HVX_ALIGN;
static int32_t bt1[ST]     HVX_ALIGN;
static int8_t  Wt2[S*ST]   HVX_ALIGN;
static int32_t bt2[S]      HVX_ALIGN;
static int8_t  Wc1[D*DFF]  HVX_ALIGN;
static int32_t bc1[DFF]    HVX_ALIGN;
static int8_t  Wc2[DFF*D]  HVX_ALIGN;
static int32_t bc2[D]      HVX_ALIGN;

static uint8_t TMh[ST*D]   HVX_ALIGN;
static int8_t  Y  [S*D]    HVX_ALIGN;
static uint8_t CMh[S*DFF]  HVX_ALIGN;
static int8_t  out[S*D]    HVX_ALIGN;
static int8_t  ref[S*D]    HVX_ALIGN;

static void mixer_ref(void) {
    /* ---- Token-mixing (mix over S, shared across channels d) ---- */
    for (int d = 0; d < D; d++) {
        for (int sp = 0; sp < ST; sp++) {
            int acc = 0;
            for (int s = 0; s < S; s++) acc += (int)X[s*D+d] * (int)Wt1[sp*S+s];
            int p = (acc >> TM_SH1) + bt1[sp];
            TMh[sp*D+d] = (uint8_t)mx_relu_i8(p);
        }
    }
    for (int d = 0; d < D; d++) {
        for (int s = 0; s < S; s++) {
            int acc2 = 0;
            for (int sp = 0; sp < ST; sp++) acc2 += (int)TMh[sp*D+d] * (int)Wt2[s*ST+sp];
            int p2 = (acc2 >> TM_SH2) + bt2[s];
            int tmo = mx_sat_i8(p2);
            Y[s*D+d] = mx_sat_i8((int)X[s*D+d] + tmo);   /* residual 1 */
        }
    }
    /* ---- Channel-mixing (mix over D, shared across tokens s) ---- */
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < DFF; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Y[s*D+d] * (int)Wc1[d*DFF+j];
            int p = (acc >> CM_SH1) + bc1[j];
            CMh[s*DFF+j] = (uint8_t)mx_relu_i8(p);
        }
        for (int d = 0; d < D; d++) {
            int acc2 = 0;
            for (int j = 0; j < DFF; j++) acc2 += (int)CMh[s*DFF+j] * (int)Wc2[j*D+d];
            int p2 = (acc2 >> CM_SH2) + bc2[d];
            int cmo = mx_sat_i8(p2);
            ref[s*D+d] = mx_sat_i8((int)Y[s*D+d] + cmo);   /* residual 2 */
        }
    }
}

int main(void) {
    uint32_t s = 0xA11CE5EDu;

    for (int i = 0; i < S*D;   i++) X[i]   = (uint8_t)(hvx_lcg(&s) % 4);
    for (int i = 0; i < ST*S;  i++) Wt1[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int i = 0; i < S*ST;  i++) Wt2[i] = (int8_t)((int)(hvx_lcg(&s) % 5) - 2);
    for (int i = 0; i < D*DFF; i++) Wc1[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int i = 0; i < DFF*D; i++) Wc2[i] = (int8_t)((int)(hvx_lcg(&s) % 5) - 2);
    for (int j = 0; j < ST;  j++) bt1[j] = (int32_t)((int)(hvx_lcg(&s) % 65)  - 32);
    for (int j = 0; j < S;   j++) bt2[j] = (int32_t)((int)(hvx_lcg(&s) % 33)  - 16);
    for (int j = 0; j < DFF; j++) bc1[j] = (int32_t)((int)(hvx_lcg(&s) % 129) - 64);
    for (int j = 0; j < D;   j++) bc2[j] = (int32_t)((int)(hvx_lcg(&s) % 33)  - 16);

    bt1[0] = -4000; bt1[1] = 4000;   /* token-mix hidden: always-kill / always-fire */
    bt2[0] = -300;  bt2[1] = 300;    /* token-mix output: sat-low / sat-high */
    bc1[0] = -4000; bc1[1] = 4000;   /* channel-mix hidden: always-kill / always-fire */
    bc2[0] = -300;  bc2[1] = 300;    /* channel-mix output: sat-low / sat-high */

    mixer_ref();

    for (int i = 0; i < S*D; i++) *((volatile signed char *)&out[i]) = (signed char)0xA5; /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, Wt1, bt1, Wt2, bt2, Wc1, bc1, Wc2, bc2, out, S, D, ST, DFF); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*D; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, S*D, fb, gotv, expv);
    return errors ? 1 : 0;
}
