#include <stdint.h>
#include <math.h>

static unsigned char v_eq[786432];
static unsigned char v_lt[786432];
static unsigned char v_isnan[786432];
static unsigned char v_logical_or[786432];
static int64_t v_scalar_tensor[1];
static int64_t v_scalar_tensor_1[1];
static int64_t v_where[786432];
static float v_where_1[786432];
static float v_relu[786432];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0) {
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_eq[i0*1024 + i1*1] = (v_xs_0[i0*1024 + i1*1] == 0);
    }
  }
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_lt[i0*1024 + i1*1] = (v_xs_0[i0*1024 + i1*1] < 0);
    }
  }
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_isnan[i0*1024 + i1*1] = (v_xs_0[i0*1024 + i1*1] != v_xs_0[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_logical_or[i0*1024 + i1*1] = (v_lt[i0*1024 + i1*1] || v_isnan[i0*1024 + i1*1]);
    }
  }
  v_scalar_tensor[0] = (int64_t)(1);
  v_scalar_tensor_1[0] = (int64_t)(0);
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_where[i0*1024 + i1*1] = (v_logical_or[i0*1024 + i1*1] ? v_scalar_tensor_1[0] : v_scalar_tensor[0]);
    }
  }
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_where_1[i0*1024 + i1*1] = (v_eq[i0*1024 + i1*1] ? v_xs_1[i0*1024 + i1*1] : v_where[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_relu[i0*1024 + i1*1] = (v_where_1[i0*1024 + i1*1] > 0 ? v_where_1[i0*1024 + i1*1] : 0);
    }
  }
  for (int i = 0; i < 786432; i++) { out0[i] = v_relu[i]; }
}
