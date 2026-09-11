/* Near-miss: ignores runtime exp_lut -- uses a hardcoded ramp table.
 * Correct grouping logic, but wrong softmax values when harness sweeps other LUT tables. */
#include <stdint.h>

#define H_Q        4
#define H_KV       2
#define GROUP_SIZE 2
#define SEQ        8
#define HEAD_DIM   16

static uint8_t HARDCODED[256];
static int lut_init = 0;
static void init_lut(void) {
    if (lut_init) return;
    for (int i = 0; i < 256; i++) HARDCODED[i] = (uint8_t)((i * 200) / 255 + 1);
    HARDCODED[255] = 200; HARDCODED[0] = 1;
    lut_init = 1;
}

static int8_t requant_i8(int32_t raw, int32_t mult, int shift) {
    int64_t v    = (int64_t)raw * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t *Q,
                      const int8_t *K,
                      const int8_t *V,
                      const uint8_t *exp_lut,   /* BUG: ignored */
                      int8_t *out,
                      int32_t scale_mult, int scale_shift,
                      int32_t av_mult,    int av_shift) {
    (void)exp_lut;
    init_lut();

    for (int hq = 0; hq < H_Q; hq++) {
        int hkv = hq / GROUP_SIZE;
        const int8_t *Qh  = Q + hq  * SEQ * HEAD_DIM;
        const int8_t *Kh  = K + hkv * SEQ * HEAD_DIM;
        const int8_t *Vh  = V + hkv * SEQ * HEAD_DIM;
        int8_t       *outh = out + hq * SEQ * HEAD_DIM;

        int8_t scores[SEQ * SEQ];
        for (int i = 0; i < SEQ; i++) {
            for (int j = 0; j < SEQ; j++) {
                int32_t acc = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    acc += (int32_t)Qh[i*HEAD_DIM+d] * (int32_t)Kh[j*HEAD_DIM+d];
                scores[i*SEQ+j] = requant_i8(acc, scale_mult, scale_shift);
            }
        }

        uint8_t probs[SEQ * SEQ];
        for (int i = 0; i < SEQ; i++) {
            const int8_t *row = scores + i * SEQ;
            uint8_t      *pr  = probs  + i * SEQ;
            int8_t m = row[0];
            for (int j = 1; j < SEQ; j++) if (row[j] > m) m = row[j];
            int32_t S = 0;
            for (int j = 0; j < SEQ; j++) {
                int diff = (int)row[j] - (int)m;
                if (diff < -255) diff = -255;
                pr[j] = HARDCODED[diff + 255];   /* BUG */
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
                outh[i*HEAD_DIM+d] = requant_i8(acc, av_mult, av_shift);
            }
        }
    }
}
