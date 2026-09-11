/* Near-miss: sums |a[i]| instead of a[i]^2. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out){
    int32_t s=0; for(int i=0;i<n;i++){ int v=a[i]<0?-(int)a[i]:(int)a[i]; s+=v; } out[0]=s;
}
