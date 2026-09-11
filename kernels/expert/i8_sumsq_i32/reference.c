#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out){
    int32_t s=0; for(int i=0;i<n;i++) s+=(int32_t)a[i]*(int32_t)a[i]; out[0]=s;
}
