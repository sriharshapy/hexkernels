#include <stdint.h>
#include <math.h>

static int32_t v_bitwise_not[6144];

extern "C" void candidate_kernel(const int32_t *v_args_0, int32_t *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_bitwise_not[i0*128 + i1*1] = (~v_args_0[i0*128 + i1*1]);
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_bitwise_not[i]; }
}
