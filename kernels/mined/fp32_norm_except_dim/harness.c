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
    "Zj+72pw/"
    ;

static const float golden_out_0[48] __attribute__((aligned(128))) = { 0x1.840de80000000p+3f, 0x1.7b5dc40000000p+3f, 0x1.6f50fc0000000p+3f, 0x1.9161720000000p+3f, 0x1.72e17a0000000p+3f, 0x1.9312480000000p+3f, 0x1.66ec140000000p+3f, 0x1.90e6900000000p+3f, 0x1.85ea8a0000000p+3f, 0x1.7c53d60000000p+3f, 0x1.8e09960000000p+3f, 0x1.7632560000000p+3f, 0x1.78ffaa0000000p+3f, 0x1.7d1ae60000000p+3f, 0x1.76ea880000000p+3f, 0x1.7569a20000000p+3f, 0x1.76d1060000000p+3f, 0x1.73cdac0000000p+3f, 0x1.7bf4840000000p+3f, 0x1.7e46be0000000p+3f, 0x1.7e98140000000p+3f, 0x1.7a22700000000p+3f, 0x1.7e7dd60000000p+3f, 0x1.7f93740000000p+3f, 0x1.76e1340000000p+3f, 0x1.88e5760000000p+3f, 0x1.79c7700000000p+3f, 0x1.8afe740000000p+3f, 0x1.711be80000000p+3f, 0x1.75aec40000000p+3f, 0x1.7978ee0000000p+3f, 0x1.859e140000000p+3f, 0x1.745d4e0000000p+3f, 0x1.7696ae0000000p+3f, 0x1.8d84e60000000p+3f, 0x1.803f8a0000000p+3f, 0x1.6b7b500000000p+3f, 0x1.887c000000000p+3f, 0x1.77801c0000000p+3f, 0x1.747a2a0000000p+3f, 0x1.782d520000000p+3f, 0x1.7cb7f60000000p+3f, 0x1.75d42e0000000p+3f, 0x1.80b1d60000000p+3f, 0x1.79bad80000000p+3f, 0x1.7531820000000p+3f, 0x1.7ad6900000000p+3f, 0x1.7cfb900000000p+3f };
static float out_0[48] __attribute__((aligned(128)));

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

    for (int i = 0; i < 48; i++) {
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
    total_n += 48;

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
