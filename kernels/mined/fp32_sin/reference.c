#include <stdint.h>
#include <math.h>

static float v_sin[196608];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sin[i0*512 + i1*1] = sinf(v_args_0[i0*512 + i1*1]);
    }
  }
  for (int i = 0; i < 196608; i++) { out0[i] = v_sin[i]; }
}
