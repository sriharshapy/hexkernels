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

static const float golden_in_0[32] __attribute__((aligned(128))) = { 0x1.486d760000000p-1f, 0x1.64a1900000000p+0f, 0x1.3245ee0000000p-1f, 0x1.a32c060000000p-1f, 0x1.a3d0a80000000p-1f, 0x1.dfb41c0000000p-1f, 0x1.b29cb20000000p-1f, 0x1.4817a40000000p-1f, 0x1.d5f5800000000p-1f, 0x1.27e7300000000p-1f, 0x1.42c7dc0000000p+0f, 0x1.61ec940000000p-1f, 0x1.9cbe4a0000000p-1f, 0x1.3015a20000000p+0f, 0x1.820a540000000p-1f, 0x1.27d0880000000p+0f, 0x1.7cd5a80000000p-1f, 0x1.3db1ec0000000p+0f, 0x1.d0f99e0000000p-1f, 0x1.5bb2380000000p+0f, 0x1.4605200000000p-1f, 0x1.5824240000000p+0f, 0x1.e7fc4a0000000p-1f, 0x1.4098bc0000000p+0f, 0x1.70c1ec0000000p+0f, 0x1.5a6caa0000000p-1f, 0x1.a15bb20000000p-1f, 0x1.5f24680000000p+0f, 0x1.0b42b80000000p+0f, 0x1.63e5b80000000p+0f, 0x1.e22d0e0000000p-1f, 0x1.b515b40000000p-1f };

static const float golden_out_0[32] __attribute__((aligned(128))) = { 0x1.486d760000000p-1f, 0x1.64a1900000000p+0f, 0x1.3245ee0000000p-1f, 0x1.a32c060000000p-1f, 0x1.a3d0a80000000p-1f, 0x1.dfb41c0000000p-1f, 0x1.b29cb20000000p-1f, 0x1.4817a40000000p-1f, 0x1.d5f5800000000p-1f, 0x1.27e7300000000p-1f, 0x1.42c7dc0000000p+0f, 0x1.61ec940000000p-1f, 0x1.9cbe4a0000000p-1f, 0x1.3015a20000000p+0f, 0x1.820a540000000p-1f, 0x1.27d0880000000p+0f, 0x1.7cd5a80000000p-1f, 0x1.3db1ec0000000p+0f, 0x1.d0f99e0000000p-1f, 0x1.5bb2380000000p+0f, 0x1.4605200000000p-1f, 0x1.5824240000000p+0f, 0x1.e7fc4a0000000p-1f, 0x1.4098bc0000000p+0f, 0x1.70c1ec0000000p+0f, 0x1.5a6caa0000000p-1f, 0x1.a15bb20000000p-1f, 0x1.5f24680000000p+0f, 0x1.0b42b80000000p+0f, 0x1.63e5b80000000p+0f, 0x1.e22d0e0000000p-1f, 0x1.b515b40000000p-1f };
static float out_0[32] __attribute__((aligned(128)));

int main() {
    forge2_hmx_enable();
    int total_errors = 0;
    int total_n = 0;
    int first_bad = -1;
    int last_bad = -1, bad_stride = -1, run_len = 0;
    int r32 = -1, r64 = -1, r128 = -1;
    int same32 = 1, same64 = 1, same128 = 1, uniform = 1;
    candidate_kernel(golden_in_0, out_0);

    for (int i = 0; i < 32; i++) {
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
    total_n += 32;

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
