/* Near-miss: broadcasts bias[r] per ROW instead of bias[c] per COLUMN (a
 * transposed broadcast-axis mistake). Compiles; wrong whenever bias values
 * differ across indices (R != C here anyway, but even where R==C this would
 * still be wrong unless bias happened to be constant). */
#include <stdint.h>
void candidate_kernel(const int32_t *acc, const int32_t *bias, int32_t *out, int R, int C){
    for (int r=0;r<R;r++) for (int c=0;c<C;c++) out[r*C+c] = acc[r*C+c] + bias[r];
}
