#include <stdint.h>
#include <math.h>

static unsigned char v_logical_xor[512];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_logical_xor[i0*64 + i1*1] = ((!!v_args_0[i0*64 + i1*1]) != (!!v_args_1[i0*64 + i1*1]));
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_logical_xor[i]; }
}
