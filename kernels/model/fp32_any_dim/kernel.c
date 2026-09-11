#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline HVX_Vector load_unaligned_128(const void *ptr) {
  HVX_Vector value;
  __builtin_memcpy(&value, ptr, 128);
  return value;
}

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  HVX_Vector acc0 =
      load_unaligned_128(static_cast<const void *>(v_args_0));
  HVX_Vector acc1 =
      load_unaligned_128(static_cast<const void *>(v_args_0 + 32));

  for (int r = 1; r < 8; ++r) {
    const float *row = v_args_0 + r * 64;
    HVX_Vector x0 =
        load_unaligned_128(static_cast<const void *>(row));
    HVX_Vector x1 =
        load_unaligned_128(static_cast<const void *>(row + 32));
    acc0 = Q6_V_vor_VV(acc0, x0);
    acc1 = Q6_V_vor_VV(acc1, x1);
  }

  uint32_t reduced[64] __attribute__((aligned(128)));
  __builtin_memcpy(reduced, &acc0, 128);
  __builtin_memcpy(reduced + 32, &acc1, 128);

  for (int i = 0; i < 64; ++i) {
    out0[i] = static_cast<unsigned char>(reduced[i] != 0);
  }
}
