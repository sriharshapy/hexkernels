/* Near-miss: computes sum of |a[i]-b[i]| (SAD / L1) instead of sum of (a[i]-b[i])^2 (L2-sq).
   For any non-zero diff != 1, |d| != d^2, so this fails. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int n, int32_t *out){
    int32_t s=0;
    for(int i=0;i<n;i++){
        int32_t d=(int32_t)a[i]-(int32_t)b[i];
        s+=(d<0)?-d:d;
    }
    out[0]=s;
}
