/* epilogue_relu_i8 harness. Small in-cache tile epilogue (no DMA/l2fetch),
 * distinct from i8_relu_l2fetch's large-N bandwidth-bound streaming variant.
 * Harness owns main(); computes the scalar reference independently. */
#include "harness_common.h"
#include "kernel_api.h"

#define N 1000   /* NOT a multiple of 128 -- real tail path, small in-cache tile */

static int8_t x[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xC0FFEEu;
    for (int i = 0; i < N; i++) x[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases. */
    x[0] = 0;      /* exact zero -> relu(0)=0 */
    x[1] = -128;   /* most negative int8 -> must clamp to 0, not wrap */
    x[2] = 127;    /* max positive -> passthrough */
    x[3] = -1;
    x[4] = 1;
    for (int i = 5;  i < 25; i++)  x[i] = -5;  /* all-negative block */
    for (int i = 25; i < 45; i++)  x[i] = 5;   /* all-positive block */
    x[N-1] = -128; /* extreme value also at the tail-path boundary */

    for (int i = 0; i < N; i++) ref[i] = (x[i] > 0) ? x[i] : 0;
    for (int i = 0; i < N; i++) *((volatile int8_t *)&out[i]) = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)out[i]; expv = (long)ref[i]; }
        }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
