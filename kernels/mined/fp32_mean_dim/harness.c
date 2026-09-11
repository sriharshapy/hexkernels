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

static float golden_in_0[4096] __attribute__((aligned(128)));
static const char b64_in_0[] =
    "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"
    "pOaHPw=="
    ;

static float golden_out_0[4096] __attribute__((aligned(128)));
static const char b64_out_0[] =
    "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"
    "pOaHPw=="
    ;
static float out_0[4096] __attribute__((aligned(128)));
static float accum_s_0[4096] __attribute__((aligned(128)));
static const char b64_acc_0[] =
    "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"
    "pOaHPw=="
    ;

int main() {
    forge2_hmx_enable();
    forge2_b64(b64_in_0, golden_in_0, sizeof(golden_in_0));
    forge2_b64(b64_out_0, golden_out_0, sizeof(golden_out_0));
    forge2_b64(b64_acc_0, accum_s_0, sizeof(accum_s_0));
    int total_errors = 0;
    int total_n = 0;
    int first_bad = -1;
    int last_bad = -1, bad_stride = -1, run_len = 0;
    int r32 = -1, r64 = -1, r128 = -1;
    int same32 = 1, same64 = 1, same128 = 1, uniform = 1;
    candidate_kernel(golden_in_0, out_0);

    for (int i = 0; i < 4096; i++) {
        if (!forge2_close_acc((float)out_0[i], (float)golden_out_0[i], accum_s_0[i], 0x1.0000000000000p-23f, 1e-4f, 1e-3f)) {
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
    total_n += 4096;

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
