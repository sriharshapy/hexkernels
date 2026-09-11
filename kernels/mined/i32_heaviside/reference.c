#include <stdint.h>
#include <math.h>

static unsigned char v_eq[98304];
static unsigned char v_lt[98304];
static unsigned char v_isnan[98304];
static unsigned char v_logical_or[98304];
static int64_t v_scalar_tensor[1];
static int64_t v_scalar_tensor_1[1];
static int64_t v_where[98304];
static int64_t v_where_1[98304];
static int32_t v__to_copy[98304];

extern "C" void candidate_kernel(const int32_t *v_args_0, const int32_t *v_args_1, int32_t *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_eq[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] == 0);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_lt[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] < 0);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_isnan[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] != v_args_0[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_logical_or[i0*384 + i1*1] = (v_lt[i0*384 + i1*1] || v_isnan[i0*384 + i1*1]);
    }
  }
  v_scalar_tensor[0] = (int64_t)(1);
  v_scalar_tensor_1[0] = (int64_t)(0);
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_where[i0*384 + i1*1] = (v_logical_or[i0*384 + i1*1] ? v_scalar_tensor_1[0] : v_scalar_tensor[0]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_where_1[i0*384 + i1*1] = (v_eq[i0*384 + i1*1] ? v_args_1[i0*384 + i1*1] : v_where[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v__to_copy[i0*384 + i1*1] = v_where_1[i0*384 + i1*1];
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v__to_copy[i]; }
}
