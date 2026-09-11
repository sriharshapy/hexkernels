/* Near-miss: sums only the data array (ignores weights entirely).
   Returns sum(a[i]) instead of sum(w[i]*a[i]). Fails when weights != 1. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *w, int n, int32_t *out){
    (void)w;  /* weights ignored */
    int32_t s=0;
    for(int i=0;i<n;i++) s+=(int32_t)a[i];
    out[0]=s;
}
