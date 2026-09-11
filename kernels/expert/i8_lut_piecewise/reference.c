#include <stdint.h>
/* Dual-table LUT: tableA for negative inputs, tableB for non-negative.
   in[i] < 0:  tableA index = (uint8_t)in[i] - 128  (maps -128->0, -1->127)
   in[i] >= 0: tableB index = in[i]                 (maps 0->0, 127->127) */
void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *tableA, const int8_t *tableB){
    for (int i = 0; i < n; i++){
        if (in[i] < 0)
            out[i] = tableA[(uint8_t)in[i] - 128];
        else
            out[i] = tableB[in[i]];
    }
}
