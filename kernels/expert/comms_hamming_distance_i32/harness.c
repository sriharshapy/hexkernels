#include "harness_common.h"
#include "kernel_api.h"
#define N 1024
static int32_t a[N]   HVX_ALIGN;
static int32_t b[N]   HVX_ALIGN;
static int32_t out[N] HVX_ALIGN;

static int32_t ref_hamming(int32_t ai, int32_t bi) {
    /* XOR then popcount (standard Hamming-weight algorithm, all 32 bits). */
    uint32_t v = (uint32_t)ai ^ (uint32_t)bi;
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;
    return (int32_t)((v * 0x01010101u) >> 24);
}

int main(void) {
    uint32_t s = 0xDEADBEEFu;
    for (int i = 0; i < N; i++) {
        a[i] = (int32_t)hvx_lcg(&s);
        b[i] = (int32_t)hvx_lcg(&s);
    }
    /* Edge cases at fixed positions */
    a[0] = b[0] = (int32_t)0x00000000u;   /* distance 0: both zero */
    a[1] = b[1] = (int32_t)0xDEADBEEFu;   /* distance 0: equal nonzero */
    a[2] = (int32_t)0x00000000u;  b[2] = (int32_t)0xFFFFFFFFu;  /* distance 32 */
    a[3] = (int32_t)0xFFFFFFFFu;  b[3] = (int32_t)0x00000000u;  /* distance 32 */
    a[4] = (int32_t)0x55555555u;  b[4] = (int32_t)0xAAAAAAAAu;  /* distance 32: all bits flip */
    a[5] = (int32_t)0xAAAAAAAAu;  b[5] = (int32_t)0x55555555u;  /* distance 32: symmetric */
    a[6] = (int32_t)0x0000FFFFu;  b[6] = (int32_t)0xFFFF0000u;  /* distance 32: half-words swap */
    a[7] = (int32_t)0x12345678u;  b[7] = (int32_t)0x12345678u;  /* distance 0: same nonzero */

    for (int i = 0; i < N; i++) out[i] = (int32_t)0xA5A5A5A5;  /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        int32_t r = ref_hamming(a[i], b[i]);
        if (out[i] != r) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)out[i]; expv = (long)r; }
        }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
