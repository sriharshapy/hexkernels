#include <stdint.h>
#include <math.h>

static float v_cummin_0[512];
static int64_t v_cummin_1[512];

extern "C" void candidate_kernel(const float *v_args_0, float *out0, int64_t *out1) {
  for (int i1 = 0; i1 < 64; i1++) {
    float best = (float)INFINITY;
    int64_t arg = 0;
    for (int s0 = 0; s0 < 8; s0++) {
      if (v_args_0[s0*64 + i1*1] < best) { best = v_args_0[s0*64 + i1*1]; arg = s0; }
      v_cummin_0[s0*64 + i1*1] = best;
      v_cummin_1[s0*64 + i1*1] = arg;
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_cummin_0[i]; }
  for (int i = 0; i < 512; i++) { out1[i] = v_cummin_1[i]; }
}
