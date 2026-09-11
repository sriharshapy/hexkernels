/* NEAR-MISS: selects b on the false branch instead of c -- a plausible
 * confusion given a>b is the predicate (out[i] = a[i]>b[i] ? a[i] :
 * b[i], i.e. an ordinary max(a,b)). Compiles fine but wrong whenever
 * b[i] != c[i] on the false-predicate lanes. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, const int8_t *c, int8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        HVX_VectorPred qt = Q6_Q_vcmp_gt_VbVb(va, vb);
        *(HVX_Vector *)(out + i) = Q6_V_vmux_QVV(qt, va, vb);   /* WRONG: selects b not c */
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}
