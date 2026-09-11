#include <stdint.h>
#include <math.h>

static float v_min_1_0[512];
static int64_t v_min_1_1[512];

extern "C" void candidate_kernel(const float *v_args_0, float *out0, int64_t *out1) {
  for (int i1 = 0; i1 < 512; i1++) {
    float best = (float)INFINITY;
    int64_t arg = 0;
    for (int r0 = 0; r0 < 512; r0++) {
      if (v_args_0[r0*512 + i1*1] < best) { best = v_args_0[r0*512 + i1*1]; arg = r0; }
    }
    v_min_1_0[i1*1] = best;
    v_min_1_1[i1*1] = arg;
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_min_1_0[i]; }
  for (int i = 0; i < 512; i++) { out1[i] = v_min_1_1[i]; }
}
