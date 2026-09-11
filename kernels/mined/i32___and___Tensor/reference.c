#include <stdint.h>
#include <math.h>

static int32_t v_bitwise_and[512];

extern "C" void candidate_kernel(const int32_t *v_args_0, const int32_t *v_args_1, int32_t *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_bitwise_and[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] & v_args_1[i0*64 + i1*1]);
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_bitwise_and[i]; }
}
