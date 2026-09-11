/* Solution 1: per-row direct broadcast-add. For each row, walk C in 32-lane
 * int32 vector chunks, re-reading `bias` directly from memory every row (no
 * staging). Row stride is C*4=400B, not 128B-aligned in general, so loads/
 * stores use the unaligned HVX_UVector form. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int32_t *acc, const int32_t *bias, int32_t *out, int R, int C) {
    const int W = 32;
    int nvec = C / W;
    for (int r = 0; r < R; r++) {
        const int32_t *arow = acc + r * C;
        int32_t *orow = out + r * C;
        int c = 0;
        for (int v = 0; v < nvec; v++, c += W) {
            HVX_Vector va = *(const HVX_UVector *)(arow + c);
            HVX_Vector vb = *(const HVX_UVector *)(bias + c);
            *(HVX_UVector *)(orow + c) = Q6_Vw_vadd_VwVw(va, vb);
        }
        for (; c < C; c++) orow[c] = arow[c] + bias[c];
    }
}
