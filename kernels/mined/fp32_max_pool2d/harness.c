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

static const float golden_in_0[256] __attribute__((aligned(128))) = { 0x1.b6bca80000000p-1f, 0x1.4a59ba0000000p-1f, 0x1.aa75900000000p-1f, 0x1.08f5100000000p+0f, 0x1.a6a17c0000000p-1f, 0x1.57fece0000000p-1f, 0x1.3192e00000000p+0f, 0x1.4a35300000000p-1f, 0x1.c6971a0000000p-1f, 0x1.85b5ca0000000p-1f, 0x1.c407ea0000000p-1f, 0x1.44afdc0000000p+0f, 0x1.46556e0000000p+0f, 0x1.5fc8100000000p+0f, 0x1.5cf2ee0000000p-1f, 0x1.69fdec0000000p-1f, 0x1.7e23aa0000000p-1f, 0x1.1d9e140000000p+0f, 0x1.156f400000000p+0f, 0x1.6601280000000p+0f, 0x1.0172ac0000000p+0f, 0x1.50ed940000000p+0f, 0x1.76bb780000000p-1f, 0x1.d8693e0000000p-1f, 0x1.1a68700000000p+0f, 0x1.de9abc0000000p-1f, 0x1.765a8c0000000p-1f, 0x1.8529140000000p-1f, 0x1.5f56300000000p+0f, 0x1.aa4b6e0000000p-1f, 0x1.0669300000000p+0f, 0x1.238cbe0000000p-1f, 0x1.0ad2c40000000p+0f, 0x1.4d50920000000p+0f, 0x1.c14f720000000p-1f, 0x1.fbbac00000000p-1f, 0x1.64d48e0000000p+0f, 0x1.1710240000000p+0f, 0x1.067e360000000p-1f, 0x1.8477280000000p-1f, 0x1.c49b9e0000000p-1f, 0x1.82d2760000000p-1f, 0x1.649a3a0000000p+0f, 0x1.8179e00000000p-1f, 0x1.08e1ce0000000p-1f, 0x1.1f43ba0000000p-1f, 0x1.f7276a0000000p-1f, 0x1.0430000000000p+0f, 0x1.3aad140000000p+0f, 0x1.7cf59c0000000p-1f, 0x1.1848d20000000p-1f, 0x1.769e080000000p+0f, 0x1.3926120000000p+0f, 0x1.fe694e0000000p-1f, 0x1.3133aa0000000p-1f, 0x1.7317740000000p-1f, 0x1.21748a0000000p-1f, 0x1.045d7e0000000p-1f, 0x1.fd48300000000p-1f, 0x1.ca148a0000000p-1f, 0x1.bd0a100000000p-1f, 0x1.6972a60000000p+0f, 0x1.6884d80000000p+0f, 0x1.9b1a740000000p-1f, 0x1.2fa43a0000000p+0f, 0x1.29438c0000000p+0f, 0x1.7805200000000p+0f, 0x1.7141cc0000000p+0f, 0x1.a87dc20000000p-1f, 0x1.b10f520000000p-1f, 0x1.f1d8c80000000p-1f, 0x1.acc54c0000000p-1f, 0x1.3e70dc0000000p+0f, 0x1.4bc6cc0000000p+0f, 0x1.5b731c0000000p+0f, 0x1.2d57780000000p+0f, 0x1.7f12800000000p-1f, 0x1.47f8640000000p+0f, 0x1.5be9900000000p-1f, 0x1.4a3cec0000000p+0f, 0x1.7537c20000000p+0f, 0x1.4089f40000000p-1f, 0x1.f0dd640000000p-1f, 0x1.8f34540000000p-1f, 0x1.2e023a0000000p-1f, 0x1.1dba080000000p+0f, 0x1.37588c0000000p+0f, 0x1.48acda0000000p+0f, 0x1.0f18940000000p+0f, 0x1.7765700000000p-1f, 0x1.31bf800000000p+0f, 0x1.3f13b20000000p+0f, 0x1.3f6a980000000p+0f, 0x1.ce2dda0000000p-1f, 0x1.0d50240000000p+0f, 0x1.23b86a0000000p-1f, 0x1.aabd2e0000000p-1f, 0x1.6fa2740000000p+0f, 0x1.7ac5840000000p+0f, 0x1.779d840000000p+0f, 0x1.61de900000000p+0f, 0x1.d378540000000p-1f, 0x1.34e3160000000p-1f, 0x1.df8ff80000000p-1f, 0x1.6e8a980000000p+0f, 0x1.01f3780000000p+0f, 0x1.0034e80000000p-1f, 0x1.db7f2e0000000p-1f, 0x1.092f6c0000000p-1f, 0x1.86ec1c0000000p-1f, 0x1.f3817a0000000p-1f, 0x1.145a0e0000000p-1f, 0x1.46c1040000000p+0f, 0x1.f635080000000p-1f, 0x1.6764580000000p-1f, 0x1.2cdc860000000p+0f, 0x1.cf5cd80000000p-1f, 0x1.c91d060000000p-1f, 0x1.013f840000000p+0f, 0x1.c141d60000000p-1f, 0x1.23a23e0000000p-1f, 0x1.44f4d00000000p+0f, 0x1.17e8e40000000p+0f, 0x1.1e37580000000p-1f, 0x1.7e7a920000000p-1f, 0x1.6e1ade0000000p+0f, 0x1.6ad5fc0000000p+0f, 0x1.f9fdca0000000p-1f, 0x1.7029880000000p+0f, 0x1.ce2c8a0000000p-1f, 0x1.95985c0000000p-1f, 0x1.120b540000000p+0f, 0x1.5677120000000p-1f, 0x1.7103900000000p+0f, 0x1.0805ea0000000p+0f, 0x1.ebb1e60000000p-1f, 0x1.e96c0e0000000p-1f, 0x1.4215e00000000p+0f, 0x1.ef482a0000000p-1f, 0x1.6a65c60000000p+0f, 0x1.9034900000000p-1f, 0x1.735caa0000000p-1f, 0x1.5ad1e60000000p+0f, 0x1.63aa960000000p+0f, 0x1.4cd3d00000000p+0f, 0x1.6e0fac0000000p+0f, 0x1.3fc4300000000p-1f, 0x1.157e740000000p+0f, 0x1.0292dc0000000p+0f, 0x1.db768a0000000p-1f, 0x1.24604c0000000p+0f, 0x1.6db5a20000000p-1f, 0x1.f558460000000p-1f, 0x1.3565bc0000000p+0f, 0x1.eb7e980000000p-1f, 0x1.0997be0000000p-1f, 0x1.2f30780000000p+0f, 0x1.0d390a0000000p+0f, 0x1.bfcaa80000000p-1f, 0x1.265f580000000p+0f, 0x1.442da60000000p+0f, 0x1.5c96900000000p-1f, 0x1.05ebb40000000p+0f, 0x1.718a820000000p-1f, 0x1.abf13c0000000p-1f, 0x1.5bde6c0000000p+0f, 0x1.1af2dc0000000p+0f, 0x1.7e3c440000000p+0f, 0x1.7ee6fa0000000p+0f, 0x1.6b6ab20000000p-1f, 0x1.f1b3920000000p-1f, 0x1.843aa80000000p-1f, 0x1.4157760000000p-1f, 0x1.7306ac0000000p+0f, 0x1.7ab63c0000000p+0f, 0x1.3e67c00000000p+0f, 0x1.67bd660000000p-1f, 0x1.088cdc0000000p+0f, 0x1.3efc640000000p-1f, 0x1.3220040000000p+0f, 0x1.53b4660000000p-1f, 0x1.7b74760000000p+0f, 0x1.0a90780000000p+0f, 0x1.49514c0000000p+0f, 0x1.8a1bc00000000p-1f, 0x1.2bb7b80000000p+0f, 0x1.3d467e0000000p+0f, 0x1.f403160000000p-1f, 0x1.93c3ec0000000p-1f, 0x1.48684a0000000p+0f, 0x1.0b45040000000p+0f, 0x1.5faa140000000p+0f, 0x1.52ec500000000p+0f, 0x1.035e860000000p-1f, 0x1.c429740000000p-1f, 0x1.4a817a0000000p+0f, 0x1.c8abd00000000p-1f, 0x1.0043580000000p+0f, 0x1.60c6080000000p+0f, 0x1.d47b140000000p-1f, 0x1.514a7c0000000p+0f, 0x1.1f32900000000p+0f, 0x1.3fb7d60000000p-1f, 0x1.7501060000000p+0f, 0x1.a5a1f00000000p-1f, 0x1.179afa0000000p+0f, 0x1.8669960000000p-1f, 0x1.f467960000000p-1f, 0x1.ee087e0000000p-1f, 0x1.cf7f1a0000000p-1f, 0x1.4cf8400000000p+0f, 0x1.9b705e0000000p-1f, 0x1.b4ca4e0000000p-1f, 0x1.1167e00000000p+0f, 0x1.2b97c80000000p+0f, 0x1.0988ac0000000p+0f, 0x1.9243ba0000000p-1f, 0x1.37953c0000000p+0f, 0x1.4f94900000000p+0f, 0x1.6b2dd40000000p+0f, 0x1.5946140000000p+0f, 0x1.50c8480000000p-1f, 0x1.1e72280000000p+0f, 0x1.e55d6a0000000p-1f, 0x1.590a120000000p+0f, 0x1.436e6c0000000p-1f, 0x1.3af9f40000000p+0f, 0x1.a8fada0000000p-1f, 0x1.1ae05a0000000p+0f, 0x1.5d89cc0000000p+0f, 0x1.09110a0000000p+0f, 0x1.59866c0000000p+0f, 0x1.71c5a20000000p-1f, 0x1.f1094e0000000p-1f, 0x1.776d0c0000000p+0f, 0x1.47a5f20000000p-1f, 0x1.02df840000000p+0f, 0x1.7d60440000000p+0f, 0x1.01fcf00000000p-1f, 0x1.3333c00000000p+0f, 0x1.45c5d00000000p+0f, 0x1.019de00000000p+0f, 0x1.3620840000000p+0f, 0x1.3441740000000p-1f, 0x1.301c400000000p-1f, 0x1.4a762c0000000p-1f, 0x1.46a6960000000p+0f, 0x1.2f49d80000000p+0f, 0x1.e304c00000000p-1f, 0x1.3dd6460000000p+0f, 0x1.7930a40000000p+0f, 0x1.e00ff60000000p-1f, 0x1.1d067e0000000p-1f, 0x1.919e160000000p-1f, 0x1.c80eaa0000000p-1f, 0x1.3cb6680000000p-1f };

static const float golden_out_0[64] __attribute__((aligned(128))) = { 0x1.c6971a0000000p-1f, 0x1.44afdc0000000p+0f, 0x1.5fc8100000000p+0f, 0x1.3192e00000000p+0f, 0x1.1d9e140000000p+0f, 0x1.6601280000000p+0f, 0x1.5f56300000000p+0f, 0x1.0669300000000p+0f, 0x1.4d50920000000p+0f, 0x1.649a3a0000000p+0f, 0x1.64d48e0000000p+0f, 0x1.0430000000000p+0f, 0x1.3aad140000000p+0f, 0x1.769e080000000p+0f, 0x1.6972a60000000p+0f, 0x1.6884d80000000p+0f, 0x1.4bc6cc0000000p+0f, 0x1.7805200000000p+0f, 0x1.47f8640000000p+0f, 0x1.4a3cec0000000p+0f, 0x1.7537c20000000p+0f, 0x1.3f13b20000000p+0f, 0x1.3f6a980000000p+0f, 0x1.48acda0000000p+0f, 0x1.6fa2740000000p+0f, 0x1.7ac5840000000p+0f, 0x1.61de900000000p+0f, 0x1.f3817a0000000p-1f, 0x1.46c1040000000p+0f, 0x1.2cdc860000000p+0f, 0x1.6e1ade0000000p+0f, 0x1.6ad5fc0000000p+0f, 0x1.7029880000000p+0f, 0x1.6a65c60000000p+0f, 0x1.7103900000000p+0f, 0x1.63aa960000000p+0f, 0x1.6e0fac0000000p+0f, 0x1.157e740000000p+0f, 0x1.2f30780000000p+0f, 0x1.265f580000000p+0f, 0x1.7ee6fa0000000p+0f, 0x1.05ebb40000000p+0f, 0x1.7306ac0000000p+0f, 0x1.7e3c440000000p+0f, 0x1.2bb7b80000000p+0f, 0x1.3d467e0000000p+0f, 0x1.7b74760000000p+0f, 0x1.5faa140000000p+0f, 0x1.52ec500000000p+0f, 0x1.7501060000000p+0f, 0x1.179afa0000000p+0f, 0x1.60c6080000000p+0f, 0x1.37953c0000000p+0f, 0x1.6b2dd40000000p+0f, 0x1.5946140000000p+0f, 0x1.2b97c80000000p+0f, 0x1.590a120000000p+0f, 0x1.776d0c0000000p+0f, 0x1.7d60440000000p+0f, 0x1.59866c0000000p+0f, 0x1.45c5d00000000p+0f, 0x1.7930a40000000p+0f, 0x1.919e160000000p-1f, 0x1.46a6960000000p+0f };
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
