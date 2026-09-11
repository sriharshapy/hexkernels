#include <stdint.h>
#include <math.h>

static unsigned char v_logical_not[6144];
static unsigned char v_any_1[1];
static unsigned char v_logical_not_1[1];

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_logical_not[i0*128 + i1*1] = (!v_args_0[i0*128 + i1*1]);
    }
  }
  {
    int32_t acc = 0;
    for (int r0 = 0; r0 < 48; r0++) {
      for (int r1 = 0; r1 < 128; r1++) {
        acc = (acc || (v_logical_not[r0*128 + r1*1] != 0));
      }
    }
    v_any_1[0] = (unsigned char)(acc);
  }
  v_logical_not_1[0] = (!v_any_1[0]);
  for (int i = 0; i < 1; i++) { out0[i] = v_logical_not_1[i]; }
}
