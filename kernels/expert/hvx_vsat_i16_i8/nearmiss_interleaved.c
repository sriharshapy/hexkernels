/* NEAR-MISS: assumes Q6_Vb_vpack_VhVh_sat INTERLEAVES like the related
 * vround instruction (out[2j]=b[j], out[2j+1]=a[j]) instead of the actual
 * CONCATENATION behavior (out[j]=b[j] for j<64, out[64+j]=a[j]). Compiles
 * and produces a fully wrong byte order for generic inputs. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int8_t sat8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n) {
    int base = 0;
    for (; base + 64 <= n; base += 64) {
        for (int j = 0; j < 64; j++) {           /* WRONG: interleaved, not concatenated */
            out[2*base + 2*j]     = sat8(b[base + j]);
            out[2*base + 2*j + 1] = sat8(a[base + j]);
        }
    }
    int m = n - base;
    for (int j = 0; j < m; j++) {
        out[2*base + 2*j]     = sat8(b[base + j]);
        out[2*base + 2*j + 1] = sat8(a[base + j]);
    }
}
