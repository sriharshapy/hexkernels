#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define NUM_HEADS   4
#define HEAD_DIM   32
#define CACHE_LEN  64
#define CACHE_SIZE (NUM_HEADS * CACHE_LEN * HEAD_DIM)   /* 8192 */
#define NEW_SIZE   (NUM_HEADS * HEAD_DIM)               /* 128 */

static int8_t  k_cache[CACHE_SIZE] HVX_ALIGN;
static int8_t  v_cache[CACHE_SIZE] HVX_ALIGN;
static int8_t  k_cache_ref[CACHE_SIZE] HVX_ALIGN;
static int8_t  v_cache_ref[CACHE_SIZE] HVX_ALIGN;
static int8_t  k_new[NEW_SIZE]     HVX_ALIGN;
static int8_t  v_new[NEW_SIZE]     HVX_ALIGN;

/*
 * Reference: indexed write into K and V caches.
 */
static void kv_cache_ref(int8_t *kc, int8_t *vc,
                          const int8_t *kn, const int8_t *vn,
                          int pos, int nh, int cl, int hd) {
    for (int h = 0; h < nh; h++) {
        int base = (h * cl + pos) * hd;
        for (int d = 0; d < hd; d++) {
            kc[base + d] = kn[h * hd + d];
            vc[base + d] = vn[h * hd + d];
        }
    }
}

/*
 * Sweep: 4 different write positions — anti-hardcode.
 * A kernel that writes to a fixed offset fails on non-matching positions.
 */
static const int POSITIONS[] = { 0, 31, 32, 63 };
#define NPOS ((int)(sizeof(POSITIONS)/sizeof(POSITIONS[0])))

int main(void) {
    uint32_t s = 0x1337C0DEu;

    /* Generate k_new/v_new (same across position sweeps; position is the variable) */
    int8_t kn_sets[NPOS][NEW_SIZE];
    int8_t vn_sets[NPOS][NEW_SIZE];
    for (int p = 0; p < NPOS; p++) {
        for (int i = 0; i < NEW_SIZE; i++) {
            kn_sets[p][i] = (int8_t)(hvx_lcg(&s) >> 24);
            vn_sets[p][i] = (int8_t)(hvx_lcg(&s) >> 24);
        }
        /* Edge values */
        kn_sets[p][0]           = 127;
        kn_sets[p][HEAD_DIM-1]  = -128;
        vn_sets[p][0]           = -128;
        vn_sets[p][NEW_SIZE-1]  = 127;
    }

    /* Generate a base cache (non-zero background to detect wrong writes) */
    int8_t base_k[CACHE_SIZE], base_v[CACHE_SIZE];
    for (int i = 0; i < CACHE_SIZE; i++) base_k[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < CACHE_SIZE; i++) base_v[i] = (int8_t)(hvx_lcg(&s) >> 24);

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int p = 0; p < NPOS; p++) {
        int pos = POSITIONS[p];

        /* Load base cache */
        for (int i = 0; i < CACHE_SIZE; i++) k_cache[i] = base_k[i];
        for (int i = 0; i < CACHE_SIZE; i++) v_cache[i] = base_v[i];
        for (int i = 0; i < CACHE_SIZE; i++) k_cache_ref[i] = base_k[i];
        for (int i = 0; i < CACHE_SIZE; i++) v_cache_ref[i] = base_v[i];

        for (int i = 0; i < NEW_SIZE; i++) k_new[i] = kn_sets[p][i];
        for (int i = 0; i < NEW_SIZE; i++) v_new[i] = vn_sets[p][i];

        /* Build reference */
        kv_cache_ref(k_cache_ref, v_cache_ref, k_new, v_new,
                     pos, NUM_HEADS, CACHE_LEN, HEAD_DIM);

        /* Run candidate */
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, {
            candidate_kernel(k_cache, v_cache, k_new, v_new,
            pos, NUM_HEADS, CACHE_LEN, HEAD_DIM);
        });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        /* Compare full caches (including positions NOT written — must be unchanged) */
        for (int i = 0; i < CACHE_SIZE; i++) {
            if (k_cache[i] != k_cache_ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = p * 2 * CACHE_SIZE + i;
                    gotv = (long)k_cache[i];
                    expv = (long)k_cache_ref[i];
                }
            }
        }
        for (int i = 0; i < CACHE_SIZE; i++) {
            if (v_cache[i] != v_cache_ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = p * 2 * CACHE_SIZE + CACHE_SIZE + i;
                    gotv = (long)v_cache[i];
                    expv = (long)v_cache_ref[i];
                }
            }
        }
    }

    hvx_report(errors, NPOS * 2 * CACHE_SIZE, fb, gotv, expv);
    return errors ? 1 : 0;
}
