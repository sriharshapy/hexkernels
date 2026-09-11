#include <stdint.h>
#include <math.h>

static float v_i0[6144];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_i0[i0*128 + i1*1] = (1.0f + ((v_args_0[i0*128 + i1*1]/3.75f)*(v_args_0[i0*128 + i1*1]/3.75f))*(3.5156229f + ((v_args_0[i0*128 + i1*1]/3.75f)*(v_args_0[i0*128 + i1*1]/3.75f))*(3.0899424f + ((v_args_0[i0*128 + i1*1]/3.75f)*(v_args_0[i0*128 + i1*1]/3.75f))*(1.2067492f + ((v_args_0[i0*128 + i1*1]/3.75f)*(v_args_0[i0*128 + i1*1]/3.75f))*(0.2659732f + ((v_args_0[i0*128 + i1*1]/3.75f)*(v_args_0[i0*128 + i1*1]/3.75f))*(0.0360768f + ((v_args_0[i0*128 + i1*1]/3.75f)*(v_args_0[i0*128 + i1*1]/3.75f))*0.0045813f))))));
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_i0[i]; }
}
