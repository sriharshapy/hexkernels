#include "harness_common.h"
#include "kernel_api.h"

#define G 98
#define N (4 * G)

static const int pattern[4] = {10, 20, 30, 40};

static uint8_t a[N] HVX_ALIGN;
static uint32_t out[G] HVX_ALIGN, ref[G] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x54Du;
    for (int i = 0; i < N; i++)
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);

    a[0]=0; a[1]=0; a[2]=0; a[3]=0;
    a[4]=10; a[5]=20; a[6]=30; a[7]=40;
    a[8]=255; a[9]=255; a[10]=255; a[11]=255;
    { int base = 4*(G-1); a[base]=5; a[base+1]=5; a[base+2]=5; a[base+3]=5; }

    for (int k = 0; k < G; k++) {
        uint32_t sad = 0;
        for (int j = 0; j < 4; j++) {
            int d = (int)a[4*k+j] - pattern[j];
            sad += (uint32_t)(d < 0 ? -d : d);
        }
        ref[k] = sad;
    }

    for (int k = 0; k < G; k++) out[k] = 0xA5A5A5A5u;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, G); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int k = 0; k < G; k++) {
        if (out[k] != ref[k]) {
            errors++;
            if (fb < 0) fb = k;
        }
    }
    hvx_report(errors, G, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
