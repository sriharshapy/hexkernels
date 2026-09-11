/* EXPERT (achievability bar) -- hvx + dma + vtcm.
 * Packs X ONCE (same k-group vrmpy layout as baseline.c), then DMA-double-
 * buffers A's rows through two VTCM row-slots: prefetch row m+1 into the
 * off-slot while row m's data is bulk-copied out of VTCM and reduced.
 *
 * IMPORTANT: a SCALAR access to a VTCM address costs a documented ~48
 * cycles/access fixed penalty in this timing model (far worse than a scalar
 * DDR access) -- so the vrmpy trick's scalar 4-byte splat reads must NEVER
 * touch the VTCM address directly. Instead, each row is bulk VECTOR-copied
 * (128B at a time -- vector loads/stores to VTCM are cheap) out of its VTCM
 * slot into a small cacheable scratch buffer, and the scalar splat
 * extraction runs against THAT (regular, cache-friendly memory). This is
 * the same "pack/unpack in cacheable buffers, move to/from VTCM in bulk
 * vector copies" pattern used for HMX crouton pack/unpack. */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define ROWBUF    8192      /* >= max K (8101), one DMA slot per A row */
#define MAX_NGROUPS 2026

static desc_t d_in, d_pf;

void candidate_kernel(const int8_t *A, const int8_t *X, int32_t *C,
                      int M, int K, int N) {
    int Kpad = (K + 3) & ~3;
    int ngroups = Kpad / 4;

    static int8_t __attribute__((aligned(128))) BT_pack[MAX_NGROUPS * 128];
    memset(BT_pack, 0, (size_t)ngroups * 128);
    for (int k = 0; k < K; k++) {
        int k4 = k / 4, r = k % 4;
        int8_t *dst = BT_pack + k4 * 128 + r;
        for (int j = 0; j < N; j++)
            dst[j * 4] = X[(size_t)k * N + j];
    }

    if (M == 0) return;

    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + ROWBUF;

    /* Cacheable scratch row buffer: the vrmpy scalar splat reads hit THIS,
     * never the VTCM address directly. */
    static int8_t __attribute__((aligned(128))) rowbuf[ROWBUF];
    const int nvec = ROWBUF / 128;

    d_in.next = 0; d_in.ctrl = (uint32_t)K;
    d_in.src = (uint32_t)(uintptr_t)A; d_in.dst = vt0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int m = 0; m < M; m++) {
        uint32_t cur = (m & 1) ? vt1 : vt0;
        uint32_t nxt = (m & 1) ? vt0 : vt1;

        if (m + 1 < M) {
            d_pf.next = 0; d_pf.ctrl = (uint32_t)K;
            d_pf.src = (uint32_t)(uintptr_t)(A + (size_t)(m + 1) * K);
            d_pf.dst = nxt;
            Q6_dmstart_A(&d_pf);
        }

        /* Bulk vector-copy this row out of VTCM into the cacheable buffer,
         * then zero the tiny k-tail beyond K (Kpad-K <= 3 bytes) so the hot
         * loop below can read full groups with NO per-iteration bounds
         * check. */
        {
            const HVX_Vector *vsrc = (const HVX_Vector *)(uintptr_t)cur;
            HVX_Vector *vdst = (HVX_Vector *)rowbuf;
            for (int v = 0; v < nvec; v++) vdst[v] = vsrc[v];
            for (int i = K; i < Kpad; i++) rowbuf[i] = 0;
        }

        const int8_t *Arow = rowbuf;
        HVX_Vector vacc = Q6_V_vzero();

        for (int k4 = 0; k4 < ngroups; k4++) {
            int kb = k4 * 4;
            uint32_t a0 = (uint8_t)Arow[kb + 0];
            uint32_t a1 = (uint8_t)Arow[kb + 1];
            uint32_t a2 = (uint8_t)Arow[kb + 2];
            uint32_t a3 = (uint8_t)Arow[kb + 3];
            uint32_t w = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
            HVX_Vector va = Q6_V_vsplat_R(w);
            HVX_Vector vb = *(const HVX_Vector *)(BT_pack + k4 * 128);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vb);
        }

        if (m + 1 < M) Q6_R_dmwait();

        int32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = vacc;
        memcpy(C + (size_t)m * N, tmp, (size_t)N * sizeof(int32_t));
    }
}
