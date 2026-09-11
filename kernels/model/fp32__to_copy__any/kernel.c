#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_xs_0, unsigned char *out0) {
  // HVX vector width is 128 bytes on the v75 target.
  const HVX_Vector signless_mask = Q6_V_vsplat_R(0x7fffffff);
  HVX_Vector acc0 = Q6_V_vzero();
  HVX_Vector acc1 = Q6_V_vzero();
  HVX_Vector acc2 = Q6_V_vzero();
  HVX_Vector acc3 = Q6_V_vzero();

  const unsigned char *src =
      reinterpret_cast<const unsigned char *>(v_xs_0);

  for (int block = 0; block < 16384; ++block) {
    const int offset = block * 512;

    HVX_Vector x0;
    HVX_Vector x1;
    HVX_Vector x2;
    HVX_Vector x3;

    __builtin_memcpy(&x0, src + offset, 128);
    __builtin_memcpy(&x1, src + offset + 128, 128);
    __builtin_memcpy(&x2, src + offset + 256, 128);
    __builtin_memcpy(&x3, src + offset + 384, 128);

    acc0 = Q6_V_vor_VV(acc0, Q6_V_vand_VV(x0, signless_mask));
    acc1 = Q6_V_vor_VV(acc1, Q6_V_vand_VV(x1, signless_mask));
    acc2 = Q6_V_vor_VV(acc2, Q6_V_vand_VV(x2, signless_mask));
    acc3 = Q6_V_vor_VV(acc3, Q6_V_vand_VV(x3, signless_mask));
  }

  HVX_Vector acc = Q6_V_vor_VV(
      Q6_V_vor_VV(acc0, acc1),
      Q6_V_vor_VV(acc2, acc3));

  unsigned char reduced[128];
  __builtin_memcpy(reduced, &acc, 128);

  unsigned char result = 0;
  for (int i = 0; i < 128; ++i) {
    if (reduced[i] != 0) {
      result = 1;
      break;
    }
  }

  out0[0] = result;
}
