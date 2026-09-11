#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>

/* H_Q=4, H_KV=2, GROUP_SIZE=2, SEQ=8, HEAD_DIM=16 */
static int8_t  Q[H_Q *SEQ*HEAD_DIM] HVX_ALIGN;
static int8_t  K[H_KV*SEQ*HEAD_DIM] HVX_ALIGN;
static int8_t  V[H_KV*SEQ*HEAD_DIM] HVX_ALIGN;
static uint8_t exp_lut[256]         HVX_ALIGN;
static int8_t  out[H_Q*SEQ*HEAD_DIM] HVX_ALIGN;
static int8_t  ref[H_Q*SEQ*HEAD_DIM] HVX_ALIGN;

static int8_t requant_i8(int32_t raw, int32_t mult, int shift) {
    int64_t v    = (int64_t)raw * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static void gqa_ref(const int8_t *Qv, const int8_t *Kv, const int8_t *Vv,
                    const uint8_t *lut, int8_t *r,
                    int32_t smult, int sshift, int32_t amult, int ashift) {
    for (int hq = 0; hq < H_Q; hq++) {
        int hkv = hq / GROUP_SIZE;   /* which KV head this query head uses */
        const int8_t *Qh = Qv + hq  * SEQ * HEAD_DIM;
        const int8_t *Kh = Kv + hkv * SEQ * HEAD_DIM;
        const int8_t *Vh = Vv + hkv * SEQ * HEAD_DIM;
        int8_t *rh = r + hq * SEQ * HEAD_DIM;

        int8_t scores[SEQ*SEQ];
        for (int i = 0; i < SEQ; i++) {
            for (int j = 0; j < SEQ; j++) {
                int32_t acc = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    acc += (int32_t)Qh[i*HEAD_DIM+d] * (int32_t)Kh[j*HEAD_DIM+d];
                scores[i*SEQ+j] = requant_i8(acc, smult, sshift);
            }
        }

        uint8_t probs[SEQ*SEQ];
        for (int i = 0; i < SEQ; i++) {
            const int8_t *row = scores + i*SEQ;
            uint8_t *pr = probs + i*SEQ;
            int8_t m = row[0];
            for (int j = 1; j < SEQ; j++) if (row[j] > m) m = row[j];
            int32_t S = 0;
            for (int j = 0; j < SEQ; j++) {
                int diff = (int)row[j] - (int)m;
                if (diff < -255) diff = -255;
                pr[j] = lut[diff + 255];
                S += (int32_t)pr[j];
            }
            int32_t half_S = S / 2;
            for (int j = 0; j < SEQ; j++) {
                int32_t ej = (int32_t)pr[j];
                pr[j] = (uint8_t)((ej * 255 + half_S) / S);
            }
        }

        for (int i = 0; i < SEQ; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int32_t acc = 0;
                for (int j = 0; j < SEQ; j++)
                    acc += (int32_t)probs[i*SEQ+j] * (int32_t)Vh[j*HEAD_DIM+d];
                rh[i*HEAD_DIM+d] = requant_i8(acc, amult, ashift);
            }
        }
    }
}

/* 2 param sets x 2 LUT tables = 4 sweeps */
static const int32_t SMULTS[]  = { 3, 1 };
static const int     SSHIFTS[] = { 7, 4 };
static const int32_t AMULTS[]  = { 5, 2 };
static const int     ASHIFTS[] = { 8, 5 };
#define NPARAMS 2
#define NLUTS   2
#define NSETS   (NPARAMS * NLUTS)

int main(void) {
    uint32_t s = 0x1A2B3C4Du;

    for (int i = 0; i < H_Q *SEQ*HEAD_DIM; i++) Q[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H_KV*SEQ*HEAD_DIM; i++) K[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H_KV*SEQ*HEAD_DIM; i++) V[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    Q[0] = 127;  K[0] = 127;
    Q[HEAD_DIM] = -128; K[HEAD_DIM] = 127;
    V[0] = 127;  V[1] = -128;
    /* Plant values in second KV head to distinguish correct vs wrong head mapping */
    for (int i = 0; i < SEQ; i++) {
        K[H_KV*SEQ*HEAD_DIM - (i+1)*HEAD_DIM] = (int8_t)(i * 10 - 40);
        V[H_KV*SEQ*HEAD_DIM - (i+1)*HEAD_DIM] = (int8_t)(i * 5  - 20);
    }

    uint8_t luts[NLUTS][256];
    for (int k = 0; k < NLUTS; k++) {
        for (int j = 0; j < 256; j++)
            luts[k][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[k][255] = 200 + k * 30;
        luts[k][0]   = 1;
    }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int li = 0; li < NLUTS; li++) {
        for (int pi = 0; pi < NPARAMS; pi++) {
            int32_t smult  = SMULTS[pi];  int sshift = SSHIFTS[pi];
            int32_t amult  = AMULTS[pi];  int ashift = ASHIFTS[pi];
            for (int j = 0; j < 256; j++) exp_lut[j] = luts[li][j];

            gqa_ref(Q, K, V, exp_lut, ref, smult, sshift, amult, ashift);
            for (int i = 0; i < H_Q*SEQ*HEAD_DIM; i++) out[i] = (int8_t)0xA5;
            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, V, exp_lut, out, smult, sshift, amult, ashift); });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            for (int i = 0; i < H_Q*SEQ*HEAD_DIM; i++) {
                if (out[i] != ref[i]) {
                    errors++;
                    if (fb < 0) {
                        fb = (li*NPARAMS + pi) * H_Q*SEQ*HEAD_DIM + i;
                        gotv = (long)out[i]; expv = (long)ref[i];
                    }
                }
            }
        }
    }

    hvx_report(errors, NSETS * H_Q*SEQ*HEAD_DIM, fb, gotv, expv);
    return errors ? 1 : 0;
}
