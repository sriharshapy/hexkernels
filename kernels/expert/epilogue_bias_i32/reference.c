#include <stdint.h>
void candidate_kernel(const int32_t *acc, const int32_t *bias, int32_t *out, int R, int C){
    for (int r=0;r<R;r++) for (int c=0;c<C;c++) out[r*C+c] = acc[r*C+c] + bias[c];
}
