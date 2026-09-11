/* Pure scalar int8 GQA QK^T (bit-exact) -- the DENOMINATOR baseline.
 * Plain nested-loop dot product (no HVX/HMX matrix engine). Each query head
 * independently computes its scores against the shared KV head via direct
 * row-row dot products (K row j IS key vector j -- no transpose needed).
 * This baseline does NOT exploit the shared-KV-head structure -- the HMX
 * expert's weight-crouton reuse across the GROUP_SIZE query heads is the
 * mechanism that must beat it by >=1.2x. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out) {
    for (int hq = 0; hq < GQA_H_Q; hq++) {
        int hkv = hq / GQA_GROUP_SIZE;
        const uint8_t *Qh = Q + (size_t)hq  * GQA_S * GQA_D;
        const int8_t  *Kh = K + (size_t)hkv * GQA_S * GQA_D;
        uint16_t      *Oh = out + (size_t)hq * GQA_S * GQA_S;
        for (int i = 0; i < GQA_S; i++)
            for (int j = 0; j < GQA_S; j++) {
                int acc = 0;
                for (int d = 0; d < GQA_D; d++) acc += (int)Qh[i*GQA_D+d] * (int)Kh[j*GQA_D+d];
                Oh[i*GQA_S+j] = (uint16_t)hvx_hmx_requant_0x40(acc);
            }
    }
}
