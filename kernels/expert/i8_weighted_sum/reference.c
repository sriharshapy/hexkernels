#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *w, int n, int32_t *out){
    int32_t s=0;
    for(int i=0;i<n;i++) s+=(int32_t)a[i]*(int32_t)w[i];
    out[0]=s;
}
