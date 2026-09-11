/* sol_05: HVX aligned vector add when segments are large, scalar for small k/seg2 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n, int k) {
    int seg1 = n - k;
    /* Segment 1: a[k..n-1] + b[0..seg1-1] -> out[0..seg1-1] */
    {
        const int8_t *pa = a + k;
        const int8_t *pb = b;
        int8_t *po = out;
        int i = 0;
        for (; i + 128 <= seg1; i += 128) {
            HVX_Vector va = *((HVX_UVector *)(pa + i));
            HVX_Vector vb = *((HVX_UVector *)(pb + i));
            *((HVX_UVector *)(po + i)) = Q6_Vb_vadd_VbVb(va, vb);
        }
        for (; i < seg1; i++)
            po[i] = (int8_t)(pa[i] + pb[i]);
    }
    /* Segment 2: a[0..k-1] + b[seg1..n-1] -> out[seg1..n-1] */
    {
        const int8_t *pa = a;
        const int8_t *pb = b + seg1;
        int8_t *po = out + seg1;
        int i = 0;
        for (; i + 128 <= k; i += 128) {
            HVX_Vector va = *((HVX_UVector *)(pa + i));
            HVX_Vector vb = *((HVX_UVector *)(pb + i));
            *((HVX_UVector *)(po + i)) = Q6_Vb_vadd_VbVb(va, vb);
        }
        for (; i < k; i++)
            po[i] = (int8_t)(pa[i] + pb[i]);
    }
}