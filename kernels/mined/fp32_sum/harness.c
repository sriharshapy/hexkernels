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
/* Tolerance for an output that IS AN ACCUMULATION.
 *
 * Scaling a float tolerance to the OUTPUT magnitude is the wrong ruler
 * for a reduction: when the terms cancel the result is near zero while
 * the error is set by the magnitude of what was summed. The standard
 * bound is |computed - exact| <= n*u*sum|terms| with u the accumulator's
 * unit roundoff, so two DIFFERENT valid summation orders differ by at
 * most 2*n*u*sum|terms|. `knu` is that 2*n*u, computed once per output;
 * `s` is this element's own sum|terms|.
 *
 * MEASURED on fp16_vecdot: the 6 elements that failed the output-scaled
 * tolerance had |want| 0.0015-0.52 against sum|terms| of 305-348, and
 * numpy's own sum failed the same 6. With this term both a sequential
 * and a pairwise summation pass 0/1024, while an fp16 ACCUMULATOR still
 * fails 84/1024 -- it discriminates, it does not merely loosen. */
static inline int forge2_close_acc(float g, float e, float s, float knu,
                                  float atol, float rtol) {
    float d = g - e; if (d < 0) d = -d;
    float ae = e < 0 ? -e : e;
    return d <= atol + rtol * ae + knu * s;
}

extern "C" void candidate_kernel(const float* in0, float* out0);

/* base64 -> bytes, for the golden buffers emitted as a byte image.
 * See oracle.B64_THRESHOLD for why they are not element literals. */
static int forge2_b64_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;                 /* '=' padding, or anything unexpected */
}
static void forge2_b64(const char *src, void *dst, unsigned long nbytes) {
    unsigned char *out = (unsigned char *)dst;
    unsigned long w = 0;
    unsigned acc = 0;
    int have = 0;
    for (const char *p = src; *p; p++) {
        int v = forge2_b64_val(*p);
        if (v < 0) continue;
        acc = (acc << 6) | (unsigned)v;
        have += 6;
        if (have >= 8) {
            have -= 8;
            if (w < nbytes) out[w++] = (unsigned char)((acc >> have) & 0xFF);
        }
    }
    /* Fail loudly rather than run against a half-filled buffer: a short
     * decode would look exactly like a wrong kernel. */
    if (w != nbytes) {
        printf("HVXENV_INCORRECT errors=-1 n=%lu first_bad=-1 (b64 decoded %lu of %lu bytes)\n", nbytes, w, nbytes);
        exit(1);
    }
}

static float golden_in_0[6144] __attribute__((aligned(128)));
static const char b64_in_0[] =
    "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"
    "g0AnP+KgmD8YoxM/dEwfP5igMz8e658//YJWP4BJmT8kP4Y/WhiePxe3ST8MBpM/53+BPx/duT+b3wg/XY2LP371dD/KoFI/agKAP7mzjT+ks1Q/ic+DPwTj"
    "hD+wK4g/jbu/P+PiTz+JjV0/J2epP6r8qj8byog/A/O2P678hD/A9Ks/WcNdPx7BPT+MZgg/hLmNP9GKjD+1K78/JFFBPyG1Fz+KAwQ/GGiMP+ahgT9EkZ0/"
    "vmBOP8QoWj+WB5c/ClOdP/6fdj/0MLo//mB0PzG6Fz8j2nQ/uGO1P6sRTD80aLY/IuybP5L1oj+LapE/9t6HP2tGhD/qTrc/M0axP1sqXj9MCm0/hlwbP3zH"
    "oz/eLZ4/gnafP4dAET9UZoQ/v/NUP4xpTD8Ll1g/cE5fPxrUWD++aYQ/UClhP3BdqT8SP1Q/hkdePz4vGD/BgLg/eFBjP5LKVD8R710/xx1ZP75vij817Sk/"
    "mmRPP3nyPz9JuD0/C3ieP2S9OD+UEX0/nzWLP5RKTz9c24k/ZdgbP99Ibj/L/zg/FNg2P7YgUD9x6aw/FjKtP10vuj+8lJM/rY8bP4Kbsz8kD1w/SLGsP5qp"
    "tD8WQqo/lKSbPxxEQj/SuSg/2gc8P3ndfz9KW5w/XiO2P0yMLj8wG6k/S30yP8fwLz+3FTU/9pM5P0fVfT9aYKo/TeFZPx69jD+G+2U/afVxPzoPhz/I3L8/"
    "Q6BLPwZ5oT+/KIY/aBabP4H1az+GZTI/mT1/P6IaTz8YGqw/9xkUPxl/Gj/gaJ0/f5mJP8YtiD+wrJQ/5JdXP87Lpj//ew4/hHmyP7Q/Oz/sLYg/U/MnP3TM"
    "uj9GB5Y/INGfP6T5iz8bDSI/smKWP1fktj+wxRk/VMOuP1uwuz9Bar8/yTWFP+pCgD9Seb4/hN4ZP1JlpT/s/5I/w1N6Pw9bWj+6NnM/1LFVP+YJKT+I9K0/"
    "BLmKPyB9SD8p3ao/wBaeP97jtz96NJc/Nql8PwD1pz8aWLE/KK8TP9aNsz+MOio/8OZ2P8UfCj+sqgU/3gOEPyO2bT8FtrI/FkwNPwZJpT+CR00/4vBjP8Tr"
    "GD9oZlM/amZPP4MuOj/rYrk/NAsfP7E7aD/QW6o/3xOgP7z2BD8pzx4/WjG+P2ipgj/MqRs/agCIP0n4tD9z9oI/uQGHPwZ3tz+alVU/4uG9P8QSgT9mEKk/"
    "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"
    "qz+oKDQ/"
    ;

static const float golden_out_0[1] __attribute__((aligned(128))) = { 0x1.80418c0000000p+12f };
static float out_0[1] __attribute__((aligned(128)));
static const float accum_s_0[1] __attribute__((aligned(128))) = { 0x1.80418c0000000p+12f };

int main() {
    forge2_hmx_enable();
    forge2_b64(b64_in_0, golden_in_0, sizeof(golden_in_0));
    int total_errors = 0;
    int total_n = 0;
    int first_bad = -1;
    int last_bad = -1, bad_stride = -1, run_len = 0;
    int r32 = -1, r64 = -1, r128 = -1;
    int same32 = 1, same64 = 1, same128 = 1, uniform = 1;
    candidate_kernel(golden_in_0, out_0);

    for (int i = 0; i < 1; i++) {
        if (!forge2_close_acc((float)out_0[i], (float)golden_out_0[i], accum_s_0[i], 0x1.8000000000000p-11f, 1e-4f, 1e-3f)) {
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
    total_n += 1;

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
