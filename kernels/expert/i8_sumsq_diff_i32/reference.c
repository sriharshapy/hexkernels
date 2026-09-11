#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int n, int32_t *out){
    int32_t s=0;
    for(int i=0;i<n;i++){
        int32_t d=(int32_t)a[i]-(int32_t)b[i];
        s+=d*d;
    }
    out[0]=s;
}
