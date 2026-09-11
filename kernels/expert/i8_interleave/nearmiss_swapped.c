/* NEARMISS: swaps the two streams -> out[2i]=b[i], out[2i+1]=a[i]. Plausible
 * (correct interleave structure) but wrong ordering. Must score INCORRECT. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    const HVX_Vector *va=(const HVX_Vector*)a, *vb=(const HVX_Vector*)b;
    HVX_Vector *vo=(HVX_Vector*)out;
    int nv = n/128, k=0;
    for (; k < nv; k++) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(va[k], vb[k], -1);  /* swapped operands */
        vo[2*k]=Q6_V_lo_W(p); vo[2*k+1]=Q6_V_hi_W(p);
    }
    for (int i = nv*128; i < n; i++) { out[2*i]=b[i]; out[2*i+1]=a[i]; }
}
