#include <stdint.h>
#include <math.h>

static float v_digamma[6144];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_digamma[i0*128 + i1*1] = (logf(v_args_0[i0*128 + i1*1] + 6.0f) - 0.5f/(v_args_0[i0*128 + i1*1] + 6.0f) + (1.0f/((v_args_0[i0*128 + i1*1] + 6.0f)*(v_args_0[i0*128 + i1*1] + 6.0f)))*(-0.0833333333f + (1.0f/((v_args_0[i0*128 + i1*1] + 6.0f)*(v_args_0[i0*128 + i1*1] + 6.0f)))*(0.0083333333f + (1.0f/((v_args_0[i0*128 + i1*1] + 6.0f)*(v_args_0[i0*128 + i1*1] + 6.0f)))*(-0.0039682540f + (1.0f/((v_args_0[i0*128 + i1*1] + 6.0f)*(v_args_0[i0*128 + i1*1] + 6.0f)))*0.0041666667f))) - 1.0f/(v_args_0[i0*128 + i1*1]) - 1.0f/(v_args_0[i0*128 + i1*1] + 1.0f) - 1.0f/(v_args_0[i0*128 + i1*1] + 2.0f) - 1.0f/(v_args_0[i0*128 + i1*1] + 3.0f) - 1.0f/(v_args_0[i0*128 + i1*1] + 4.0f) - 1.0f/(v_args_0[i0*128 + i1*1] + 5.0f));
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_digamma[i]; }
}
