#include <stdint.h>
#include <math.h>

static unsigned char v_gt[512];
static float v_mul[512];
static float v_where[512];
static float v_amax[1];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_gt[i0*64 + i1*1] = (v_xs_0[i0*64 + i1*1] > 0);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_xs_1[i0*64 + i1*1] * v_xs_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_where[i0*64 + i1*1] = (v_gt[i0*64 + i1*1] ? v_xs_0[i0*64 + i1*1] : v_mul[i0*64 + i1*1]);
    }
  }
  {
    float acc = -INFINITY;
    for (int r0 = 0; r0 < 8; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (v_where[r0*64 + r1*1] > acc ? v_where[r0*64 + r1*1] : acc);
      }
    }
    v_amax[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_amax[i]; }
}
