#include <stdint.h>
#include <math.h>

static float v__pdist_forward[28];
static unsigned char v_logical_not[28];
static unsigned char v_any_1[1];
static unsigned char v_logical_not_1[1];

extern "C" void candidate_kernel(const float *v_xs_0, unsigned char *out0) {
  int pd_k = 0;
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = i0 + 1; i1 < 8; i1++) {
      float acc = 0;
      for (int r0 = 0; r0 < 64; r0++) {
        float d = v_xs_0[i0*64 + r0] - v_xs_0[i1*64 + r0];
        acc += d * d;
      }
      v__pdist_forward[pd_k++] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 28; i0++) {
    v_logical_not[i0*1] = (!v__pdist_forward[i0*1]);
  }
  {
    int32_t acc = 0;
    for (int r0 = 0; r0 < 28; r0++) {
      acc = (acc || (v_logical_not[r0*1] != 0));
    }
    v_any_1[0] = (unsigned char)(acc);
  }
  v_logical_not_1[0] = (!v_any_1[0]);
  for (int i = 0; i < 1; i++) { out0[i] = v_logical_not_1[i]; }
}
