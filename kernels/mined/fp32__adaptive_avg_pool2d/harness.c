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

static const float golden_in_0[256] __attribute__((aligned(128))) = { 0x1.1906bc0000000p+0f, 0x1.0776880000000p+0f, 0x1.9e5eea0000000p-1f, 0x1.2ef1080000000p+0f, 0x1.15501e0000000p+0f, 0x1.3977c40000000p-1f, 0x1.3cfd180000000p+0f, 0x1.a9a0220000000p-1f, 0x1.1aac700000000p+0f, 0x1.2feaa00000000p+0f, 0x1.76a2440000000p-1f, 0x1.8e6dde0000000p-1f, 0x1.d9f5b40000000p-1f, 0x1.f87df80000000p-1f, 0x1.94a62c0000000p-1f, 0x1.4ea5ae0000000p+0f, 0x1.739a740000000p+0f, 0x1.078f6a0000000p+0f, 0x1.4212700000000p-1f, 0x1.5aea140000000p+0f, 0x1.b31b420000000p-1f, 0x1.1e8c780000000p-1f, 0x1.24efb80000000p+0f, 0x1.b34aba0000000p-1f, 0x1.7c9a9c0000000p-1f, 0x1.97c97a0000000p-1f, 0x1.1f6dd60000000p+0f, 0x1.11d5280000000p+0f, 0x1.4f33b60000000p-1f, 0x1.58e6240000000p+0f, 0x1.32bcb00000000p+0f, 0x1.0340880000000p+0f, 0x1.78b17a0000000p+0f, 0x1.3838bc0000000p+0f, 0x1.7f635a0000000p-1f, 0x1.9512920000000p-1f, 0x1.08e7800000000p+0f, 0x1.43c2be0000000p+0f, 0x1.d1c1160000000p-1f, 0x1.46b7740000000p+0f, 0x1.3098c80000000p+0f, 0x1.ac135c0000000p-1f, 0x1.13726c0000000p-1f, 0x1.4f19d00000000p+0f, 0x1.30baf60000000p+0f, 0x1.041c480000000p+0f, 0x1.f33dca0000000p-1f, 0x1.6f41d60000000p+0f, 0x1.0744600000000p+0f, 0x1.22e7900000000p+0f, 0x1.59cb620000000p-1f, 0x1.5e95940000000p+0f, 0x1.ad3c3e0000000p-1f, 0x1.3cbd1e0000000p+0f, 0x1.5a396c0000000p-1f, 0x1.6cdefa0000000p+0f, 0x1.9b32da0000000p-1f, 0x1.7fb6180000000p-1f, 0x1.90f52e0000000p-1f, 0x1.7fa3b40000000p+0f, 0x1.8a2abc0000000p-1f, 0x1.1de6640000000p+0f, 0x1.3b50a20000000p-1f, 0x1.7c7f7e0000000p+0f, 0x1.3aafac0000000p+0f, 0x1.52e8420000000p-1f, 0x1.2d37380000000p+0f, 0x1.e7c58c0000000p-1f, 0x1.5e20cc0000000p+0f, 0x1.8553d40000000p-1f, 0x1.3dc3540000000p+0f, 0x1.3475ee0000000p-1f, 0x1.30a6ca0000000p+0f, 0x1.cfe80c0000000p-1f, 0x1.2310d40000000p+0f, 0x1.a3f64a0000000p-1f, 0x1.fa34280000000p-1f, 0x1.1ed1680000000p-1f, 0x1.6614f40000000p+0f, 0x1.36987c0000000p+0f, 0x1.842a4e0000000p-1f, 0x1.aaaf960000000p-1f, 0x1.2739760000000p-1f, 0x1.6e3d580000000p+0f, 0x1.d0cea60000000p-1f, 0x1.48d1300000000p+0f, 0x1.7341b80000000p-1f, 0x1.24612c0000000p+0f, 0x1.8e40fc0000000p-1f, 0x1.5dec660000000p-1f, 0x1.0745540000000p-1f, 0x1.385f800000000p+0f, 0x1.7266340000000p-1f, 0x1.6f21540000000p-1f, 0x1.15a2820000000p+0f, 0x1.73024c0000000p+0f, 0x1.3e0c9e0000000p+0f, 0x1.1310100000000p-1f, 0x1.59c42e0000000p-1f, 0x1.33d6de0000000p-1f, 0x1.d108380000000p-1f, 0x1.567f480000000p-1f, 0x1.146ab00000000p+0f, 0x1.bbe3d20000000p-1f, 0x1.611b940000000p+0f, 0x1.49459c0000000p-1f, 0x1.27a96e0000000p+0f, 0x1.2162e60000000p-1f, 0x1.2713de0000000p+0f, 0x1.a4e24a0000000p-1f, 0x1.dff3360000000p-1f, 0x1.74fa820000000p+0f, 0x1.6ce7be0000000p-1f, 0x1.ce16880000000p-1f, 0x1.463bee0000000p+0f, 0x1.dd98b20000000p-1f, 0x1.de65320000000p-1f, 0x1.5d94340000000p-1f, 0x1.3197c20000000p+0f, 0x1.250ea80000000p+0f, 0x1.ba4b1a0000000p-1f, 0x1.895ad00000000p-1f, 0x1.68a9e00000000p-1f, 0x1.4556240000000p+0f, 0x1.3225000000000p+0f, 0x1.5890d00000000p+0f, 0x1.034a240000000p+0f, 0x1.6a49ea0000000p+0f, 0x1.1205100000000p-1f, 0x1.656a280000000p-1f, 0x1.55a07c0000000p+0f, 0x1.f08fb40000000p-1f, 0x1.0f07580000000p+0f, 0x1.2bea8a0000000p-1f, 0x1.a49ee20000000p-1f, 0x1.44a86c0000000p+0f, 0x1.389ed40000000p+0f, 0x1.3e05140000000p+0f, 0x1.e4e4060000000p-1f, 0x1.44eda40000000p+0f, 0x1.d951920000000p-1f, 0x1.7379c20000000p-1f, 0x1.7b20780000000p+0f, 0x1.67c95a0000000p-1f, 0x1.59b7400000000p+0f, 0x1.52cf080000000p+0f, 0x1.06f3080000000p+0f, 0x1.260afa0000000p-1f, 0x1.57d58a0000000p-1f, 0x1.6eac960000000p-1f, 0x1.c38d520000000p-1f, 0x1.58c2600000000p+0f, 0x1.47c04c0000000p+0f, 0x1.59636c0000000p+0f, 0x1.0cbfde0000000p-1f, 0x1.0d37180000000p-1f, 0x1.600a7e0000000p-1f, 0x1.6b29d00000000p+0f, 0x1.2d0cfc0000000p-1f, 0x1.b6be6c0000000p-1f, 0x1.b1ca660000000p-1f, 0x1.27733a0000000p-1f, 0x1.da55c40000000p-1f, 0x1.f658200000000p-1f, 0x1.bbbf460000000p-1f, 0x1.ef8e220000000p-1f, 0x1.135d0a0000000p+0f, 0x1.1d15ea0000000p-1f, 0x1.0986800000000p-1f, 0x1.70381e0000000p-1f, 0x1.47d3500000000p+0f, 0x1.b488ec0000000p-1f, 0x1.18291c0000000p-1f, 0x1.867faa0000000p-1f, 0x1.68994c0000000p-1f, 0x1.0603520000000p+0f, 0x1.29dd140000000p+0f, 0x1.65757c0000000p+0f, 0x1.4647b40000000p+0f, 0x1.3323dc0000000p+0f, 0x1.243f180000000p+0f, 0x1.9682fc0000000p-1f, 0x1.5f25c80000000p+0f, 0x1.6655d40000000p-1f, 0x1.8085c80000000p-1f, 0x1.2bd4120000000p+0f, 0x1.75d39c0000000p+0f, 0x1.1c9cf60000000p-1f, 0x1.38f46a0000000p-1f, 0x1.0977660000000p-1f, 0x1.891d560000000p-1f, 0x1.2e168c0000000p+0f, 0x1.4c4f1c0000000p+0f, 0x1.3936aa0000000p+0f, 0x1.039afc0000000p-1f, 0x1.5c5a200000000p+0f, 0x1.512e820000000p+0f, 0x1.2049680000000p-1f, 0x1.62e2c40000000p-1f, 0x1.6538500000000p+0f, 0x1.2ed0f80000000p+0f, 0x1.64cedc0000000p+0f, 0x1.ffa2a60000000p-1f, 0x1.13dcb40000000p+0f, 0x1.7ea0d00000000p+0f, 0x1.ad0cda0000000p-1f, 0x1.db8af40000000p-1f, 0x1.bb0d9a0000000p-1f, 0x1.fac5180000000p-1f, 0x1.6e2e7c0000000p+0f, 0x1.4ed5a40000000p+0f, 0x1.866e980000000p-1f, 0x1.17e3100000000p+0f, 0x1.3b7a4c0000000p-1f, 0x1.3c4ab80000000p-1f, 0x1.7682980000000p+0f, 0x1.15b93e0000000p-1f, 0x1.1cbb400000000p-1f, 0x1.4525d60000000p+0f, 0x1.f596d80000000p-1f, 0x1.1e9dc60000000p+0f, 0x1.081fc60000000p-1f, 0x1.9aada00000000p-1f, 0x1.2fa8900000000p+0f, 0x1.49edfa0000000p+0f, 0x1.4b4ce20000000p-1f, 0x1.4bec5e0000000p+0f, 0x1.161ff80000000p+0f, 0x1.5898f60000000p+0f, 0x1.1290000000000p-1f, 0x1.293b640000000p+0f, 0x1.a450540000000p-1f, 0x1.2520460000000p+0f, 0x1.484ab20000000p-1f, 0x1.6d15fc0000000p+0f, 0x1.6977a40000000p+0f, 0x1.32e3b00000000p-1f, 0x1.49a3860000000p-1f, 0x1.3454020000000p+0f, 0x1.32cd620000000p-1f, 0x1.604db20000000p-1f, 0x1.07ef420000000p+0f, 0x1.75eb080000000p+0f, 0x1.10de260000000p+0f, 0x1.5ffcdc0000000p+0f, 0x1.589a7e0000000p+0f, 0x1.ba01280000000p-1f, 0x1.e79f0e0000000p-1f, 0x1.c51c9e0000000p-1f, 0x1.afba9a0000000p-1f, 0x1.43e83e0000000p+0f, 0x1.5f6aec0000000p+0f, 0x1.7de6a00000000p-1f, 0x1.69fb760000000p-1f, 0x1.de69b00000000p-1f, 0x1.7969a80000000p+0f };

static const float golden_out_0[64] __attribute__((aligned(128))) = { 0x1.1ac5140000000p+0f, 0x1.c054480000000p-1f, 0x1.cda2ea0000000p-1f, 0x1.0ab17c0000000p+0f, 0x1.0156fc0000000p+0f, 0x1.0b4d920000000p+0f, 0x1.b4a9ee0000000p-1f, 0x1.0d24940000000p+0f, 0x1.2de32c0000000p+0f, 0x1.b186fe0000000p-1f, 0x1.2060600000000p+0f, 0x1.261e2e0000000p+0f, 0x1.dbd0340000000p-1f, 0x1.14e6640000000p+0f, 0x1.fb2b7e0000000p-1f, 0x1.0d08e00000000p+0f, 0x1.fe5f500000000p-1f, 0x1.05897e0000000p+0f, 0x1.d6a6c00000000p-1f, 0x1.1d2af00000000p+0f, 0x1.86c1d20000000p-1f, 0x1.deee200000000p-1f, 0x1.d0fe240000000p-1f, 0x1.19a9b40000000p+0f, 0x1.e6a9840000000p-1f, 0x1.7f94340000000p-1f, 0x1.c6a4620000000p-1f, 0x1.15d42e0000000p+0f, 0x1.9fa90c0000000p-1f, 0x1.0bacd80000000p+0f, 0x1.0a2ca00000000p+0f, 0x1.310e9e0000000p+0f, 0x1.d92dc20000000p-1f, 0x1.2152000000000p+0f, 0x1.a5b1220000000p-1f, 0x1.117f400000000p+0f, 0x1.536a800000000p+0f, 0x1.537a000000000p-1f, 0x1.bf38100000000p-1f, 0x1.d637600000000p-1f, 0x1.54bf100000000p-1f, 0x1.029bae0000000p+0f, 0x1.927d8a0000000p-1f, 0x1.ae1bfc0000000p-1f, 0x1.1eda620000000p+0f, 0x1.1f636a0000000p+0f, 0x1.885b400000000p-1f, 0x1.013d7a0000000p+0f, 0x1.4649680000000p+0f, 0x1.f8ead20000000p-1f, 0x1.0d9e9e0000000p+0f, 0x1.f0fafa0000000p-1f, 0x1.c265a40000000p-1f, 0x1.147f8c0000000p+0f, 0x1.ac26f00000000p-1f, 0x1.0469d40000000p+0f, 0x1.dc6d040000000p-1f, 0x1.4e267e0000000p+0f, 0x1.90124a0000000p-1f, 0x1.e48f200000000p-1f, 0x1.b940dc0000000p-1f, 0x1.4a87180000000p+0f, 0x1.0b22180000000p+0f, 0x1.0e5ba80000000p+0f };
static float out_0[64] __attribute__((aligned(128)));

int main() {
    forge2_hmx_enable();
    int total_errors = 0;
    int total_n = 0;
    int first_bad = -1;
    int last_bad = -1, bad_stride = -1, run_len = 0;
    int r32 = -1, r64 = -1, r128 = -1;
    int same32 = 1, same64 = 1, same128 = 1, uniform = 1;
    candidate_kernel(golden_in_0, out_0);

    for (int i = 0; i < 64; i++) {
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
    total_n += 64;

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
