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


extern "C" void candidate_kernel(const float* in0, unsigned char* out0);

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
    "AD+G4YE/"
    ;

static const unsigned char golden_out_0[1] __attribute__((aligned(128))) = { 1 };
static unsigned char out_0[1] __attribute__((aligned(128)));

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
        if (out_0[i] != golden_out_0[i]) {
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
