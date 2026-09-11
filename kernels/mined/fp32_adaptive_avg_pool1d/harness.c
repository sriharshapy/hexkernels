#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>

/* ---- everything this harness needs, emitted rather than included ----
 * Self-contained on purpose: no header from hexbench/env/harness/ is on
 * this compile's include path, so the R&D material cannot reach a forge v2
 * translation unit even by accident. See oracle.harness_c's docstring. */

/* Enable the HMX extension context: SSR.XE (bit 29) + SSR.XA=2 (bits
 * 27:25). The standalone runtime enables HVX only, so without this an
 * `mxmem` access raises exception 0x18. The program runs privileged. */
static inline void forge2_hmx_enable(void) {
    unsigned m = (1u << 29) | (2u << 25);  /* 0x24000000 */
    unsigned t;
    __asm__ volatile("%0=ssr\n\t %0=or(%0,%1)\n\t ssr=%0\n\t isync\n\t"
                     : "=&r"(t) : "r"(m));
}

/* Tolerance compare for FLOATING-POINT results. HVX float arithmetic
 * goes through the non-IEEE qfloat path and reductions reorder, so a
 * correct vectorised kernel is not bit-identical to the IEEE scalar
 * golden. Each dtype is judged at ~1-2 ULP OF ITSELF; integer results
 * stay exact and get no tolerance at all. */
static inline int forge2_close_f32(float g, float e) {
    float d = g - e; if (d < 0) d = -d;
    float ae = e < 0 ? -e : e;
    return d <= 1e-4f + 1e-3f * ae;
}

extern "C" void candidate_kernel(const float* in0, float* out0);

static const float golden_in_0[32] __attribute__((aligned(128))) = { 0x1.3b247a0000000p+0f, 0x1.6b83220000000p-1f, 0x1.1e9dbc0000000p+0f, 0x1.0bbb160000000p+0f, 0x1.d7e6440000000p-1f, 0x1.1f8c800000000p+0f, 0x1.195a4a0000000p-1f, 0x1.6102fc0000000p+0f, 0x1.1fc0900000000p+0f, 0x1.047ae20000000p-1f, 0x1.bc00c20000000p-1f, 0x1.29291e0000000p+0f, 0x1.40bafa0000000p-1f, 0x1.1cf1540000000p+0f, 0x1.5c53100000000p+0f, 0x1.ee31fa0000000p-1f, 0x1.b774060000000p-1f, 0x1.2fc1e20000000p+0f, 0x1.2390300000000p+0f, 0x1.35ec1c0000000p+0f, 0x1.611c440000000p-1f, 0x1.17f5dc0000000p-1f, 0x1.2f60600000000p+0f, 0x1.149e9a0000000p+0f, 0x1.20d3b00000000p+0f, 0x1.185b6c0000000p+0f, 0x1.110cf80000000p+0f, 0x1.7b2e7a0000000p+0f, 0x1.e893740000000p-1f, 0x1.778d9c0000000p+0f, 0x1.05c4360000000p-1f, 0x1.c8a5200000000p-1f };

static const float golden_out_0[16] __attribute__((aligned(128))) = { 0x1.f0e60c0000000p-1f, 0x1.152c680000000p+0f, 0x1.05bfd00000000p+0f, 0x1.edb0200000000p-1f, 0x1.a1fe000000000p-1f, 0x1.0394c00000000p+0f, 0x1.bd4ed00000000p-1f, 0x1.29b6060000000p+0f, 0x1.05bdf20000000p+0f, 0x1.2cbe260000000p+0f, 0x1.3c89100000000p-1f, 0x1.21ff7c0000000p+0f, 0x1.1c978e0000000p+0f, 0x1.461db80000000p+0f, 0x1.35ebac0000000p+0f, 0x1.6734ac0000000p-1f };
static float out_0[16] __attribute__((aligned(128)));

int main() {
    forge2_hmx_enable();
    int total_errors = 0;
    int total_n = 0;
    int first_bad = -1;
    int last_bad = -1, bad_stride = -1, run_len = 0;
    int r32 = -1, r64 = -1, r128 = -1;
    int same32 = 1, same64 = 1, same128 = 1, uniform = 1;
    candidate_kernel(golden_in_0, out_0);

    for (int i = 0; i < 16; i++) {
        if (!forge2_close_f32((float)out_0[i], (float)golden_out_0[i])) {
            int bi = total_n + i;
            if (first_bad < 0) {
                first_bad = bi;
                r32 = bi & 31; r64 = bi & 63; r128 = bi & 127;
                run_len = 1;
            } else {
                int gap = bi - last_bad;
                if (bad_stride < 0) bad_stride = gap;
                else if (gap != bad_stride) uniform = 0;
                if (gap == 1 && run_len == bi - first_bad) run_len++;
                if ((bi & 31)  != r32)  same32 = 0;
                if ((bi & 63)  != r64)  same64 = 0;
                if ((bi & 127) != r128) same128 = 0;
            }
            last_bad = bi;
            total_errors++;
        }
    }
    total_n += 16;

    if (total_errors == 0) {
        printf("HVXENV_CORRECT errors=0 n=%d\n", total_n);
        return 0;
    } else {
        /* the SHAPE of the bad set: what names the bug.
         * mod32/mod64/mod128 are the shared residue when EVERY
         * bad index has one, and -1 when they do not. */
        printf("HVXENV_INCORRECT errors=%d n=%d first_bad=%d last_bad=%d bad_stride=%d uniform_stride=%d run_len=%d mod32=%d mod64=%d mod128=%d\n",
               total_errors, total_n, first_bad, last_bad,
               bad_stride, (bad_stride > 0 ? uniform : 0),
               run_len,
               (same32 ? r32 : -1), (same64 ? r64 : -1),
               (same128 ? r128 : -1));
        return 1;
    }
}
