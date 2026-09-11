#include <stdint.h>
#include <math.h>

static unsigned char v_logical_or[786432];

extern "C" void candidate_kernel(const int32_t *v_args_0, const int32_t *v_args_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_logical_or[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] || v_args_1[i0*1024 + i1*1]);
    }
  }
  for (int i = 0; i < 786432; i++) { out0[i] = v_logical_or[i]; }
}
