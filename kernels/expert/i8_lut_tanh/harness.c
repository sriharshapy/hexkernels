/*
 * Harness for i8_lut_tanh.
 * Anti-cheat: runs the kernel against TWO independently-seeded random LUT tables.
 * A kernel that hardcodes one fixed table will pass sweep A but fail sweep B.
 * Edge inputs -128, 0, 127, -1 exercise all index-boundary cases.
 */
#include "harness_common.h"
#include "kernel_api.h"
#define N 1024

static int8_t in[N]    HVX_ALIGN;
static int8_t out[N]   HVX_ALIGN;
static int8_t ref[N]   HVX_ALIGN;
static int8_t lutA[256] HVX_ALIGN;  /* sweep A */
static int8_t lutB[256] HVX_ALIGN;  /* sweep B -- distinct random table */

int main(void) {
    uint32_t s = 0x4D2F13u;

    /* Build inputs (shared across both sweeps). */
    for (int i = 0; i < N; i++) in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Plant boundary-index sentinels. */
    in[0] = -128;  /* unsigned index 128 */
    in[1] =  127;  /* unsigned index 127 */
    in[2] =    0;  /* unsigned index   0 */
    in[3] =   -1;  /* unsigned index 255 */

    /* Build LUT A. */
    for (int k = 0; k < 256; k++) lutA[k] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Build LUT B (different seed epoch -> different values). */
    for (int k = 0; k < 256; k++) lutB[k] = (int8_t)(hvx_lcg(&s) >> 24);

    int total_errors = 0;
    int fb_idx = -1;
    long fb_got = 0, fb_exp = 0;

    /* --- Sweep A --- */
    for (int i = 0; i < N; i++) ref[i] = lutA[(uint8_t)in[i]];
    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;  /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, lutA); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            total_errors++;
            if (fb_idx < 0) { fb_idx = i; fb_got = (long)out[i]; fb_exp = (long)ref[i]; }
        }
    }

    /* --- Sweep B --- */
    for (int i = 0; i < N; i++) ref[i] = lutB[(uint8_t)in[i]];
    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;  /* poison */
    unsigned long long _hvx_kc_b = 0;
    HVX_TIME_KERNEL(_hvx_kc_b, { candidate_kernel(in, out, N, lutB); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc_b);
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            total_errors++;
            if (fb_idx < 0) { fb_idx = i; fb_got = (long)out[i]; fb_exp = (long)ref[i]; }
        }
    }

    hvx_report(total_errors, N * 2, fb_idx, fb_got, fb_exp);
    return total_errors ? 1 : 0;
}
