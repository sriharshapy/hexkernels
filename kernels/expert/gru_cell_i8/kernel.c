/* EXPERT (achievability bar) — HVX-vectorized integer GRU cell.
 * The three gate matvecs (Wz,Wr,Wn each [H x C], C=I+H=48) dominate. Each output
 * row j is dot(W[j,0:C], concat[0:C]). We build a 128-byte concat vector padded
 * with zeros beyond C, then per row do one unaligned 128-byte load of W[j,:] and
 * Q6_Vw_vrmpy_VbVb (signed 4-way dot) -> 32 word partial sums whose total is the
 * full C-length dot (bytes beyond C multiply the zero-padded concat = 0). The
 * cheap per-element gate/blend/requant/LUT stages (H=24) stay scalar. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

static inline int32_t requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return r;
}
static inline int32_t clamp8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}

/* dot(W[j,0:C], concat) using a padded 128-byte concat vector. */
static inline int32_t dot_row(const int8_t *wrow, HVX_Vector cvec) {
    HVX_Vector wv = *(const HVX_UVector *)wrow;           /* unaligned 128B load */
    HVX_Vector p  = Q6_Vw_vrmpy_VbVb(wv, cvec);           /* 32 word 4-way dots */
    int32_t t[32] __attribute__((aligned(128)));
    *(HVX_Vector *)t = p;
    int32_t s = 0;
    for (int m = 0; m < 32; m++) s += t[m];
    return s;
}

void candidate_kernel(
    const int8_t *Wz, const int8_t *Wr, const int8_t *Wn,
    const int32_t *bz, const int32_t *br, const int32_t *bn,
    const int8_t *x, const int8_t *h_prev, int8_t *h_t,
    int H, int I, int32_t mult, int shift, int8_t zp,
    const int8_t *sig_lut, const int8_t *tanh_lut) {
    int C = I + H;
    int8_t z[64], r[64], rh[64], n[64];

    /* concat_xh = [x(0:I), h_prev(0:H), 0...] padded to 128 bytes. */
    int8_t cbuf[128] __attribute__((aligned(128)));
    memset(cbuf, 0, 128);
    for (int k = 0; k < I; k++) cbuf[k]     = x[k];
    for (int k = 0; k < H; k++) cbuf[I + k] = h_prev[k];
    HVX_Vector cxh = *(const HVX_Vector *)cbuf;

    /* Update gate z and reset gate r (both over concat_xh). */
    for (int j = 0; j < H; j++) {
        int32_t acc = bz[j] + dot_row(Wz + j * C, cxh);
        z[j] = sig_lut[(uint8_t)requant(acc, mult, shift, zp)];
    }
    for (int j = 0; j < H; j++) {
        int32_t acc = br[j] + dot_row(Wr + j * C, cxh);
        r[j] = sig_lut[(uint8_t)requant(acc, mult, shift, zp)];
    }

    /* Gated recurrent rh[j] = clamp(((r[j]+128)*h_prev[j] + 64) >> 7). */
    for (int j = 0; j < H; j++)
        rh[j] = (int8_t)clamp8(((int32_t)(r[j] + 128) * (int32_t)h_prev[j] + 64) >> 7);

    /* concat_xrh = [x(0:I), rh(0:H), 0...] for the candidate matvec. */
    for (int k = 0; k < H; k++) cbuf[I + k] = rh[k];
    HVX_Vector cxrh = *(const HVX_Vector *)cbuf;
    for (int j = 0; j < H; j++) {
        int32_t acc = bn[j] + dot_row(Wn + j * C, cxrh);
        n[j] = tanh_lut[(uint8_t)requant(acc, mult, shift, zp)];
    }

    /* Output blend. */
    for (int j = 0; j < H; j++) {
        int32_t blend = ((int32_t)(128 - z[j]) * (int32_t)h_prev[j]
                       + (int32_t)(z[j] + 128) * (int32_t)n[j] + 128) >> 8;
        h_t[j] = (int8_t)clamp8(blend);
    }
}
