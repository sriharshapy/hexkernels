/*
 * k05: u8_box3x3 â€” sliding window HVX.
 *
 * Precomputes HVX h-sums for all rows (even/odd streams separately),
 * then uses a sliding vertical accumulator with Q6_Vh_vsub_VhVh + Q6_Vh_vadd_VhVh
 * to update the 3-row vertical sum as we advance through output rows.
 * Final output via scalar div9.
 *
 * Diversity: uses Q6_Vh_vsub_VhVh (subtract) for sliding window update.
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#define W 130
#define H 98

static inline int cli(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static uint8_t  g_rb[256]     __attribute__((aligned(128)));
static uint16_t g_he[H][64]   __attribute__((aligned(128)));
static uint16_t g_ho[H][64]   __attribute__((aligned(128)));
static uint16_t g_tmp_e[64]   __attribute__((aligned(128)));
static uint16_t g_tmp_o[64]   __attribute__((aligned(128)));
static uint8_t  g_out[128]    __attribute__((aligned(128)));

static void compute_hrow(const uint8_t *src, int r) {
    *(HVX_Vector *)(g_rb)       = Q6_V_vzero();
    *(HVX_Vector *)(g_rb + 128) = Q6_V_vzero();
    memcpy(g_rb, src, W);

    HVX_Vector A = *(const HVX_Vector *)(g_rb);
    HVX_Vector B = *(const HVX_Vector *)(g_rb + 128);
    HVX_Vector v0 = A;
    HVX_Vector v1 = Q6_V_valign_VVR(B, A, 1);
    HVX_Vector v2 = Q6_V_valign_VVR(B, A, 2);

    HVX_VectorPair w0 = Q6_Wuh_vzxt_Vub(v0);
    HVX_VectorPair w1 = Q6_Wuh_vzxt_Vub(v1);
    HVX_VectorPair w2 = Q6_Wuh_vzxt_Vub(v2);

    *(HVX_Vector *)g_he[r] = Q6_Vh_vadd_VhVh(
        Q6_Vh_vadd_VhVh(Q6_V_lo_W(w0), Q6_V_lo_W(w1)), Q6_V_lo_W(w2));
    *(HVX_Vector *)g_ho[r] = Q6_Vh_vadd_VhVh(
        Q6_Vh_vadd_VhVh(Q6_V_hi_W(w0), Q6_V_hi_W(w1)), Q6_V_hi_W(w2));
}

void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h) {
    /* Precompute h-sums for all rows */
    for (int r = 0; r < h; r++)
        compute_hrow(in + r*w, r);

    /* Border rows: scalar */
    for (int x = 0; x < w; x++) {
        int s = 0;
        for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++)
            s += in[cli(0+dy,0,h-1)*w + cli(x+dx,0,w-1)];
        out[x] = (uint8_t)(s/9);
    }
    for (int x = 0; x < w; x++) {
        int s = 0;
        for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++)
            s += in[cli(h-1+dy,0,h-1)*w + cli(x+dx,0,w-1)];
        out[(h-1)*w + x] = (uint8_t)(s/9);
    }

    /* Init accumulator for y=1: rows 0+1+2 */
    HVX_Vector ace = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(
        *(HVX_Vector *)g_he[0], *(HVX_Vector *)g_he[1]), *(HVX_Vector *)g_he[2]);
    HVX_Vector aco = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(
        *(HVX_Vector *)g_ho[0], *(HVX_Vector *)g_ho[1]), *(HVX_Vector *)g_ho[2]);

    for (int y = 1; y < h-1; y++) {
        /* Border cols: scalar */
        { int s=0; for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) s+=in[cli(y+dy,0,h-1)*w+cli(dx,0,w-1)]; out[y*w]=(uint8_t)(s/9); }
        { int s=0; for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) s+=in[cli(y+dy,0,h-1)*w+cli(w-1+dx,0,w-1)]; out[y*w+w-1]=(uint8_t)(s/9); }

        /* Write output using current accumulator */
        *(HVX_Vector *)g_tmp_e = ace;
        *(HVX_Vector *)g_tmp_o = aco;

        for (int j = 0; j < 64; j++) {
            g_out[2*j]   = (uint8_t)(g_tmp_e[j] / 9);
            g_out[2*j+1] = (uint8_t)(g_tmp_o[j] / 9);
        }
        memcpy(out + y*w + 1, g_out, w-2);

        /* Slide: subtract row y-1, add row y+2 (if valid) */
        if (y + 2 < h) {
            ace = Q6_Vh_vadd_VhVh(
                Q6_Vh_vsub_VhVh(ace, *(HVX_Vector *)g_he[y-1]),
                *(HVX_Vector *)g_he[y+2]);
            aco = Q6_Vh_vadd_VhVh(
                Q6_Vh_vsub_VhVh(aco, *(HVX_Vector *)g_ho[y-1]),
                *(HVX_Vector *)g_ho[y+2]);
        }
    }
}