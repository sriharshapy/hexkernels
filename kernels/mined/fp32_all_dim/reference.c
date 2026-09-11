#include <stdint.h>
#include <math.h>

static unsigned char v_logical_not[2097152];
static unsigned char v_any_1[2048];
static unsigned char v_logical_not_1[2048];

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 2048; i1++) {
      v_logical_not[i0*2048 + i1*1] = (!v_args_0[i0*2048 + i1*1]);
    }
  }
  for (int i1 = 0; i1 < 2048; i1++) {
    {
      int32_t acc = 0;
      for (int r0 = 0; r0 < 1024; r0++) {
        acc = (acc || (v_logical_not[r0*2048 + i1*1] != 0));
      }
      v_any_1[i1*1] = (unsigned char)(acc);
    }
  }
  for (int i0 = 0; i0 < 2048; i0++) {
    v_logical_not_1[i0*1] = (!v_any_1[i0*1]);
  }
  for (int i = 0; i < 2048; i++) { out0[i] = v_logical_not_1[i]; }
}
