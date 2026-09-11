#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static uint8_t in[N] HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

/* Multiple thresholds so a candidate MUST read `thresh` (no hardcoding 128).
 * Includes extremes 0 (always false) and 255 (always true except ==255).
 * Also 1 (only 0 passes), 200, 64. */
static const uint8_t THRESHES[] = { 128, 0, 1, 200, 255, 64 };
#define NT ((int)(sizeof(THRESHES)/sizeof(THRESHES[0])))

int main(void) {
    uint32_t s = 0xA3C17Fu;
    for (int i = 0; i < N; i++) in[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    /* Inject exact boundary hits: in[i]==thresh must produce 0 (strict <). */
    in[0] = 128; in[1] = 127; in[2] = 129;
    in[3] = 0;   in[4] = 255; in[5] = 200;
    in[6] = 1;   in[7] = 64;
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int k = 0; k < NT; k++) {
        uint8_t thresh = THRESHES[k];
        for (int i = 0; i < N; i++) ref[i] = (in[i] < thresh) ? 255 : 0;
        for (int i = 0; i < N; i++) out[i] = 0xA5;   /* poison */
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, thresh); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i = 0; i < N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = k * N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, N * NT, fb, gotv, expv);
    return errors ? 1 : 0;
}
