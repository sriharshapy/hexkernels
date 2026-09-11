#include <stdint.h>
#include <math.h>

static unsigned char v_any_1[64];

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  for (int i1 = 0; i1 < 64; i1++) {
    {
      int32_t acc = 0;
      for (int r0 = 0; r0 < 8; r0++) {
        acc = (acc || (v_args_0[r0*64 + i1*1] != 0));
      }
      v_any_1[i1*1] = (unsigned char)(acc);
    }
  }
  for (int i = 0; i < 64; i++) { out0[i] = v_any_1[i]; }
}
