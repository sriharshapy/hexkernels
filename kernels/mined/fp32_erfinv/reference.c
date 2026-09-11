#include <stdint.h>
#include <math.h>

static float v_erfinv[6144];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_erfinv[i0*128 + i1*1] = (((((((((2.81022636e-08f*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + 3.43273939e-07f)*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + -3.5233877e-06f)*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + -4.39150654e-06f)*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + 0.00021858087f)*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + -0.00125372503f)*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + -0.00417768164f)*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + 0.246640727f)*(-logf((1.0f - v_args_0[i0*128 + i1*1])*(1.0f + v_args_0[i0*128 + i1*1])) - 2.5f) + 1.50140941f) * v_args_0[i0*128 + i1*1]);
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_erfinv[i]; }
}
