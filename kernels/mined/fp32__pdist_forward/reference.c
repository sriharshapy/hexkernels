#include <stdint.h>
#include <math.h>

static float v__pdist_forward[1128];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  int pd_k = 0;
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = i0 + 1; i1 < 48; i1++) {
      float acc = 0;
      for (int r0 = 0; r0 < 128; r0++) {
        float d = v_args_0[i0*128 + r0] - v_args_0[i1*128 + r0];
        acc += d * d;
      }
      v__pdist_forward[pd_k++] = sqrtf(acc);
    }
  }
  for (int i = 0; i < 1128; i++) { out0[i] = v__pdist_forward[i]; }
}
