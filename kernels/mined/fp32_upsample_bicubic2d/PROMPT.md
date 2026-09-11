Accelerate the kernel `fp32_upsample_bicubic2d` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v__to_copy[2097152];
static int64_t v_arange[64];
static float v__to_copy_1[64];
static int64_t v_arange_1[64];
static float v__to_copy_2[64];
static float v_add[64];
static float v_mul[64];
static float v_sub[64];
static float v_add_1[64];
static float v_mul_1[64];
static float v_sub_1[64];
static float v_unsqueeze[64];
static float v_floor[64];
static float v_floor_1[64];
static float v_sub_2[64];
static float v_clamp[64];
static float v_sub_3[64];
static float v_clamp_1[64];
static int64_t v__to_copy_3[64];
static int64_t v__to_copy_4[64];
static int64_t v_sub_4[64];
static int64_t v_add_2[64];
static int64_t v_add_3[64];
static int64_t v_sub_5[64];
static int64_t v_add_4[64];
static int64_t v_add_5[64];
static float v_sub_6[64];
static float v_cat[128];
static float v_view[128];
static float v_add_6[64];
static float v_sub_7[64];
static float v_cat_1[128];
static float v_view_1[128];
static float v_mul_2[128];
static float v_sub_8[128];
static float v_mul_3[128];
static float v_add_7[128];
static float v_mul_4[128];
static float v_sub_9[128];
static float v_mul_5[128];
static float v_sub_10[128];
static float v_mul_6[128];
static float v_mul_7[128];
static float v_add_8[128];
static float v_slice_1[64];
static float v_slice_2[64];
static float v_squeeze[64];
static float v_squeeze_1[64];
static float v_slice_3[64];
static float v_slice_4[64];
static float v_squeeze_2[64];
static float v_squeeze_3[64];
static float v_sub_11[64];
static float v_cat_2[128];
static float v_view_2[128];
static float v_add_9[64];
static float v_sub_12[64];
static float v_cat_3[128];
static float v_view_3[128];
static float v_mul_8[128];
static float v_sub_13[128];
static float v_mul_9[128];
static float v_add_10[128];
static float v_mul_10[128];
static float v_sub_14[128];
static float v_mul_11[128];
static float v_sub_15[128];
static float v_mul_12[128];
static float v_mul_13[128];
static float v_add_11[128];
static float v_slice_5[64];
static float v_slice_6[64];
static float v_squeeze_4[64];
static float v_squeeze_5[64];
static float v_slice_7[64];
static float v_slice_8[64];
static float v_squeeze_6[64];
static float v_squeeze_7[64];
static int64_t v_clamp_2[64];
static int64_t v_clamp_3[64];
static float v_index[524288];
static int64_t v_clamp_4[64];
static int64_t v_clamp_5[64];
static float v_index_1[524288];
static int64_t v_clamp_6[64];
static int64_t v_clamp_7[64];
static float v_index_2[524288];
static int64_t v_clamp_8[64];
static int64_t v_clamp_9[64];
static float v_index_3[524288];
static float v_mul_14[524288];
static float v_mul_15[524288];
static float v_add_12[524288];
static float v_mul_16[524288];
static float v_add_13[524288];
static float v_mul_17[524288];
static float v_add_14[524288];
static int64_t v_clamp_10[64];
static int64_t v_clamp_11[64];
static float v_index_4[524288];
static int64_t v_clamp_12[64];
static int64_t v_clamp_13[64];
static float v_index_5[524288];
static int64_t v_clamp_14[64];
static int64_t v_clamp_15[64];
static float v_index_6[524288];
static int64_t v_clamp_16[64];
static int64_t v_clamp_17[64];
static float v_index_7[524288];
static float v_mul_18[524288];
static float v_mul_19[524288];
static float v_add_15[524288];
static float v_mul_20[524288];
static float v_add_16[524288];
static float v_mul_21[524288];
static float v_add_17[524288];
static int64_t v_clamp_18[64];
static int64_t v_clamp_19[64];
static float v_index_8[524288];
static int64_t v_clamp_20[64];
static int64_t v_clamp_21[64];
static float v_index_9[524288];
static int64_t v_clamp_22[64];
static int64_t v_clamp_23[64];
static float v_index_10[524288];
static int64_t v_clamp_24[64];
static int64_t v_clamp_25[64];
static float v_index_11[524288];
static float v_mul_22[524288];
static float v_mul_23[524288];
static float v_add_18[524288];
static float v_mul_24[524288];
static float v_add_19[524288];
static float v_mul_25[524288];
static float v_add_20[524288];
static int64_t v_clamp_26[64];
static int64_t v_clamp_27[64];
static float v_index_12[524288];
static int64_t v_clamp_28[64];
static int64_t v_clamp_29[64];
static float v_index_13[524288];
static int64_t v_clamp_30[64];
static int64_t v_clamp_31[64];
static float v_index_14[524288];
static int64_t v_clamp_32[64];
static int64_t v_clamp_33[64];
static float v_index_15[524288];
static float v_mul_26[524288];
static float v_mul_27[524288];
static float v_add_21[524288];
static float v_mul_28[524288];
static float v_add_22[524288];
static float v_mul_29[524288];
static float v_add_23[524288];
static float v_mul_30[524288];
static float v_mul_31[524288];
static float v_add_24[524288];
static float v_mul_32[524288];
static float v_add_25[524288];
static float v_mul_33[524288];
static float v_add_26[524288];
static float v__to_copy_5[524288];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 128; i2++) {
        for (int i3 = 0; i3 < 128; i3++) {
          v__to_copy[i1*16384 + i2*128 + i3*1] = v_args_0[i1*16384 + i2*128 + i3*1];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_arange[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v__to_copy_1[i0*1] = v_arange[i0*1];
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_arange_1[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v__to_copy_2[i0*1] = v_arange_1[i0*1];
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_add[i0*1] = (v__to_copy_2[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_mul[i0*1] = (v_add[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_sub[i0*1] = (v_mul[i0*1] - 0.5f);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_add_1[i0*1] = (v__to_copy_1[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_mul_1[i0*1] = (v_add_1[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_sub_1[i0*1] = (v_mul_1[i0*1] - 0.5f);
  }
  for (int i = 0; i < 64; i++) v_unsqueeze[i] = v_sub_1[i];
  for (int i0 = 0; i0 < 64; i0++) {
    v_floor[i0*1] = floorf(v_sub[i0*1]);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_floor_1[i0*1] = floorf(v_unsqueeze[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_sub_2[i0*1] = (v_unsqueeze[i0*1] - v_floor_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp[i0*1] = ((v_sub_2[i0*1] < 0.0f ? 0.0f : v_sub_2[i0*1]) > 1.0f ? 1.0f : (v_sub_2[i0*1] < 0.0f ? 0.0f : v_sub_2[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_sub_3[i0*1] = (v_sub[i0*1] - v_floor[i0*1]);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_1[i0*1] = ((v_sub_3[i0*1] < 0.0f ? 0.0f : v_sub_3[i0*1]) > 1.0f ? 1.0f : (v_sub_3[i0*1] < 0.0f ? 0.0f : v_sub_3[i0*1]));
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v__to_copy_3[i0*1] = v_floor[i0*1];
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v__to_copy_4[i0*1] = v_floor_1[i0*1];
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_sub_4[i0*1] = (v__to_copy_4[i0*1] - 1);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_add_2[i0*1] = (v__to_copy_4[i0*1] + 1);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_add_3[i0*1] = (v__to_copy_4[i0*1] + 2);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_sub_5[i0*1] = (v__to_copy_3[i0*1] - 1);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_add_4[i0*1] = (v__to_copy_3[i0*1] + 1);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_add_5[i0*1] = (v__to_copy_3[i0*1] + 2);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_sub_6[i0*1] = (1.0f - v_clamp_1[i0*1]);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_cat[(i0)] = v_clamp_1[i0*1];
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_cat[(i0 + 64)] = v_sub_6[i0*1];
  }
  for (int i = 0; i < 128; i++) v_view[i] = v_cat[i];
  for (int i0 = 0; i0 < 64; i0++) {
    v_add_6[i0*1] = (v_clamp_1[i0*1] + 1.0f);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_sub_7[i0*1] = (2.0f - v_clamp_1[i0*1]);
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_cat_1[(i0)] = v_add_6[i0*1];
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_cat_1[(i0 + 64)] = v_sub_7[i0*1];
  }
  for (int i = 0; i < 128; i++) v_view_1[i] = v_cat_1[i];
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul_2[i0*64 + i1*1] = (v_view_1[i0*64 + i1*1] * -0.75f);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_8[i0*64 + i1*1] = (v_mul_2[i0*64 + i1*1] - -3.75f);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul_3[i0*64 + i1*1] = (v_sub_8[i0*64 + i1*1] * v_view_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_add_7[i0*64 + i1*1] = (v_mul_3[i0*64 + i1*1] + -6.0f);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul_4[i0*64 + i1*1] = (v_add_7[i0*64 + i1*1] * v_view_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_9[i0*64 + i1*1] = (v_mul_4[i0*64 + i1*1] - -3.0f);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul_5[i0*64 + i1*1] = (v_view[i0*64 + i1*1] * 1.25f);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_10[i0*64 + i1*1] = (v_mul_5[i0*64 + i1*1] - 2.25f);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul_6[i0*64 + i1*1] = (v_sub_10[i0*64 + i1*1] * v_view[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul_7[i0*64 + i1*1] = (v_mul_6[i0*64 + i1*1] * v_view[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_add_8[i0*64 + i1*1] = (v_mul_7[i0*64 + i1*1] + 1);
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_slice_1[i1*1] = v_sub_9[i0*64 + i1];
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_slice_2[i1*1] = v_sub_9[(1 + i0 * 1)*64 + i1];
    }
  }
  for (int i = 0; i < 64; i++) v_squeeze[i] = v_slice_1[i];
  for (int i = 0; i < 64; i++) v_squeeze_1[i] = v_slice_2[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_slice_3[i1*1] = v_add_8[i0*64 + i1];
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_slice_4[i1*1] = v_add_8[(1 + i0 * 1)*64 + i1];
    }
  }
  for (int i = 0; i < 64; i++) v_squeeze_2[i] = v_slice_3[i];
  for (int i = 0; i < 64; i++) v_squeeze_3[i] = v_slice_4[i];
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_sub_11[i0*1] = (1.0f - v_clamp[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_2[(i0) + (i1)] = v_clamp[i0*1];
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_2[(i0 + 64) + (i1)] = v_sub_11[i0*1];
    }
  }
  for (int i = 0; i < 128; i++) v_view_2[i] = v_cat_2[i];
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_add_9[i0*1] = (v_clamp[i0*1] + 1.0f);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_sub_12[i0*1] = (2.0f - v_clamp[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_3[(i0) + (i1)] = v_add_9[i0*1];
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_3[(i0 + 64) + (i1)] = v_sub_12[i0*1];
    }
  }
  for (int i = 0; i < 128; i++) v_view_3[i] = v_cat_3[i];
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_mul_8[i0*64 + i1*1] = (v_view_3[i0*64 + i1*1] * -0.75f);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_sub_13[i0*64 + i1*1] = (v_mul_8[i0*64 + i1*1] - -3.75f);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_mul_9[i0*64 + i1*1] = (v_sub_13[i0*64 + i1*1] * v_view_3[i0*64 + i1*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_add_10[i0*64 + i1*1] = (v_mul_9[i0*64 + i1*1] + -6.0f);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_mul_10[i0*64 + i1*1] = (v_add_10[i0*64 + i1*1] * v_view_3[i0*64 + i1*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_sub_14[i0*64 + i1*1] = (v_mul_10[i0*64 + i1*1] - -3.0f);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_mul_11[i0*64 + i1*1] = (v_view_2[i0*64 + i1*1] * 1.25f);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_sub_15[i0*64 + i1*1] = (v_mul_11[i0*64 + i1*1] - 2.25f);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_mul_12[i0*64 + i1*1] = (v_sub_15[i0*64 + i1*1] * v_view_2[i0*64 + i1*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_mul_13[i0*64 + i1*1] = (v_mul_12[i0*64 + i1*1] * v_view_2[i0*64 + i1*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_add_11[i0*64 + i1*1] = (v_mul_13[i0*64 + i1*1] + 1);
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_slice_5[i1*1] = v_sub_14[i0*64 + i1 + i2];
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_slice_6[i1*1] = v_sub_14[(1 + i0 * 1)*64 + i1 + i2];
      }
    }
  }
  for (int i = 0; i < 64; i++) v_squeeze_4[i] = v_slice_5[i];
  for (int i = 0; i < 64; i++) v_squeeze_5[i] = v_slice_6[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_slice_7[i1*1] = v_add_11[i0*64 + i1 + i2];
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_slice_8[i1*1] = v_add_11[(1 + i0 * 1)*64 + i1 + i2];
      }
    }
  }
  for (int i = 0; i < 64; i++) v_squeeze_6[i] = v_slice_7[i];
  for (int i = 0; i < 64; i++) v_squeeze_7[i] = v_slice_8[i];
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_2[i0*1] = ((v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]) > 127 ? 127 : (v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_3[i0*1] = ((v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]) > 127 ? 127 : (v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_2[(i2)])*128 + (v_clamp_3[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_4[i0*1] = ((v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]) > 127 ? 127 : (v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_5[i0*1] = ((v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]) > 127 ? 127 : (v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_1[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_4[(i2)])*128 + (v_clamp_5[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_6[i0*1] = ((v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]) > 127 ? 127 : (v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_7[i0*1] = ((v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]) > 127 ? 127 : (v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_2[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_6[(i2)])*128 + (v_clamp_7[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_8[i0*1] = ((v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]) > 127 ? 127 : (v_sub_4[i0*1] < 0 ? 0 : v_sub_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_9[i0*1] = ((v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]) > 127 ? 127 : (v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_3[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_8[(i2)])*128 + (v_clamp_9[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_14[i1*4096 + i2*64 + i3*1] = (v_index[i1*4096 + i2*64 + i3*1] * v_squeeze[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_15[i1*4096 + i2*64 + i3*1] = (v_index_1[i1*4096 + i2*64 + i3*1] * v_squeeze_2[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_12[i1*4096 + i2*64 + i3*1] = (v_mul_14[i1*4096 + i2*64 + i3*1] + v_mul_15[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_16[i1*4096 + i2*64 + i3*1] = (v_index_2[i1*4096 + i2*64 + i3*1] * v_squeeze_3[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_13[i1*4096 + i2*64 + i3*1] = (v_add_12[i1*4096 + i2*64 + i3*1] + v_mul_16[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_17[i1*4096 + i2*64 + i3*1] = (v_index_3[i1*4096 + i2*64 + i3*1] * v_squeeze_1[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_14[i1*4096 + i2*64 + i3*1] = (v_add_13[i1*4096 + i2*64 + i3*1] + v_mul_17[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_10[i0*1] = ((v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]) > 127 ? 127 : (v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_11[i0*1] = ((v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]) > 127 ? 127 : (v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_4[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_10[(i2)])*128 + (v_clamp_11[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_12[i0*1] = ((v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]) > 127 ? 127 : (v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_13[i0*1] = ((v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]) > 127 ? 127 : (v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_5[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_12[(i2)])*128 + (v_clamp_13[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_14[i0*1] = ((v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]) > 127 ? 127 : (v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_15[i0*1] = ((v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]) > 127 ? 127 : (v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_6[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_14[(i2)])*128 + (v_clamp_15[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_16[i0*1] = ((v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]) > 127 ? 127 : (v__to_copy_4[i0*1] < 0 ? 0 : v__to_copy_4[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_17[i0*1] = ((v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]) > 127 ? 127 : (v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_7[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_16[(i2)])*128 + (v_clamp_17[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_18[i1*4096 + i2*64 + i3*1] = (v_index_4[i1*4096 + i2*64 + i3*1] * v_squeeze[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_19[i1*4096 + i2*64 + i3*1] = (v_index_5[i1*4096 + i2*64 + i3*1] * v_squeeze_2[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_15[i1*4096 + i2*64 + i3*1] = (v_mul_18[i1*4096 + i2*64 + i3*1] + v_mul_19[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_20[i1*4096 + i2*64 + i3*1] = (v_index_6[i1*4096 + i2*64 + i3*1] * v_squeeze_3[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_16[i1*4096 + i2*64 + i3*1] = (v_add_15[i1*4096 + i2*64 + i3*1] + v_mul_20[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_21[i1*4096 + i2*64 + i3*1] = (v_index_7[i1*4096 + i2*64 + i3*1] * v_squeeze_1[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_17[i1*4096 + i2*64 + i3*1] = (v_add_16[i1*4096 + i2*64 + i3*1] + v_mul_21[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_18[i0*1] = ((v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]) > 127 ? 127 : (v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_19[i0*1] = ((v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]) > 127 ? 127 : (v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_8[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_18[(i2)])*128 + (v_clamp_19[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_20[i0*1] = ((v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]) > 127 ? 127 : (v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_21[i0*1] = ((v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]) > 127 ? 127 : (v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_9[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_20[(i2)])*128 + (v_clamp_21[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_22[i0*1] = ((v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]) > 127 ? 127 : (v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_23[i0*1] = ((v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]) > 127 ? 127 : (v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_10[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_22[(i2)])*128 + (v_clamp_23[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_24[i0*1] = ((v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]) > 127 ? 127 : (v_add_2[i0*1] < 0 ? 0 : v_add_2[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_25[i0*1] = ((v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]) > 127 ? 127 : (v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_11[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_24[(i2)])*128 + (v_clamp_25[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_22[i1*4096 + i2*64 + i3*1] = (v_index_8[i1*4096 + i2*64 + i3*1] * v_squeeze[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_23[i1*4096 + i2*64 + i3*1] = (v_index_9[i1*4096 + i2*64 + i3*1] * v_squeeze_2[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_18[i1*4096 + i2*64 + i3*1] = (v_mul_22[i1*4096 + i2*64 + i3*1] + v_mul_23[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_24[i1*4096 + i2*64 + i3*1] = (v_index_10[i1*4096 + i2*64 + i3*1] * v_squeeze_3[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_19[i1*4096 + i2*64 + i3*1] = (v_add_18[i1*4096 + i2*64 + i3*1] + v_mul_24[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_25[i1*4096 + i2*64 + i3*1] = (v_index_11[i1*4096 + i2*64 + i3*1] * v_squeeze_1[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_20[i1*4096 + i2*64 + i3*1] = (v_add_19[i1*4096 + i2*64 + i3*1] + v_mul_25[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_26[i0*1] = ((v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]) > 127 ? 127 : (v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_27[i0*1] = ((v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]) > 127 ? 127 : (v_sub_5[i0*1] < 0 ? 0 : v_sub_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_12[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_26[(i2)])*128 + (v_clamp_27[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_28[i0*1] = ((v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]) > 127 ? 127 : (v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_29[i0*1] = ((v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]) > 127 ? 127 : (v__to_copy_3[i0*1] < 0 ? 0 : v__to_copy_3[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_13[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_28[(i2)])*128 + (v_clamp_29[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_30[i0*1] = ((v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]) > 127 ? 127 : (v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_31[i0*1] = ((v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]) > 127 ? 127 : (v_add_4[i0*1] < 0 ? 0 : v_add_4[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_14[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_30[(i2)])*128 + (v_clamp_31[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_32[i0*1] = ((v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]) > 127 ? 127 : (v_add_3[i0*1] < 0 ? 0 : v_add_3[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    v_clamp_33[i0*1] = ((v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]) > 127 ? 127 : (v_add_5[i0*1] < 0 ? 0 : v_add_5[i0*1]));
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_index_15[i1*4096 + i2*64 + i3*1] = v__to_copy[(i0)*2097152 + (i1)*16384 + (v_clamp_32[(i2)])*128 + (v_clamp_33[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_26[i1*4096 + i2*64 + i3*1] = (v_index_12[i1*4096 + i2*64 + i3*1] * v_squeeze[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_27[i1*4096 + i2*64 + i3*1] = (v_index_13[i1*4096 + i2*64 + i3*1] * v_squeeze_2[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_21[i1*4096 + i2*64 + i3*1] = (v_mul_26[i1*4096 + i2*64 + i3*1] + v_mul_27[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_28[i1*4096 + i2*64 + i3*1] = (v_index_14[i1*4096 + i2*64 + i3*1] * v_squeeze_3[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_22[i1*4096 + i2*64 + i3*1] = (v_add_21[i1*4096 + i2*64 + i3*1] + v_mul_28[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_29[i1*4096 + i2*64 + i3*1] = (v_index_15[i1*4096 + i2*64 + i3*1] * v_squeeze_1[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_23[i1*4096 + i2*64 + i3*1] = (v_add_22[i1*4096 + i2*64 + i3*1] + v_mul_29[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_30[i1*4096 + i2*64 + i3*1] = (v_add_14[i1*4096 + i2*64 + i3*1] * v_squeeze_4[i2*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_31[i1*4096 + i2*64 + i3*1] = (v_add_17[i1*4096 + i2*64 + i3*1] * v_squeeze_6[i2*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_24[i1*4096 + i2*64 + i3*1] = (v_mul_30[i1*4096 + i2*64 + i3*1] + v_mul_31[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_32[i1*4096 + i2*64 + i3*1] = (v_add_20[i1*4096 + i2*64 + i3*1] * v_squeeze_7[i2*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_25[i1*4096 + i2*64 + i3*1] = (v_add_24[i1*4096 + i2*64 + i3*1] + v_mul_32[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_mul_33[i1*4096 + i2*64 + i3*1] = (v_add_23[i1*4096 + i2*64 + i3*1] * v_squeeze_5[i2*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v_add_26[i1*4096 + i2*64 + i3*1] = (v_add_25[i1*4096 + i2*64 + i3*1] + v_mul_33[i1*4096 + i2*64 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        for (int i3 = 0; i3 < 64; i3++) {
          v__to_copy_5[i1*4096 + i2*64 + i3*1] = v_add_26[i1*4096 + i2*64 + i3*1];
        }
      }
    }
  }
  for (int i = 0; i < 524288; i++) { out0[i] = v__to_copy_5[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d2, 0)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @main(%arg0: tensor<1x128x128x128xf32>) -> tensor<1x128x64x64xf32> {
  %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<64xi64>) {
    ^bb0(%out: i64):
    %117 = linalg.index 0 : index
    %118 = arith.index_cast %117 : index to i64
    linalg.yield %118 : i64
    } -> tensor<64xi64>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<64xi64>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: i64, %out: f32):
    %117 = arith.sitofp %in : i64 to f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.addf %in, %cst : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%4 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.mulf %in, %cst_0 : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%5 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %in, %cst : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%6 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = math.floor %in : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<64x1xf32>) outs(%8 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = math.floor %in : f32
    linalg.yield %117 : f32
    } -> tensor<64x1xf32>
  %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded, %9 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%8 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.subf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<64x1xf32>
  %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<64x1xf32>) outs(%8 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.cmpf ult, %in, %cst_1 : f32
    %118 = arith.select %117, %cst_1, %in : f32
    %119 = arith.cmpf ugt, %118, %cst_2 : f32
    %120 = arith.select %119, %cst_2, %118 : f32
    linalg.yield %120 : f32
    } -> tensor<64x1xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %7 : tensor<64xf32>, tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.subf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.cmpf ult, %in, %cst_1 : f32
    %118 = arith.select %117, %cst_1, %in : f32
    %119 = arith.cmpf ugt, %118, %cst_2 : f32
    %120 = arith.select %119, %cst_2, %118 : f32
    linalg.yield %120 : f32
    } -> tensor<64xf32>
  %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%7 : tensor<64xf32>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: f32, %out: i64):
    %117 = arith.fptosi %in : f32 to i64
    linalg.yield %117 : i64
    } -> tensor<64xi64>
  %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<64x1xf32>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: f32, %out: i64):
    %117 = arith.fptosi %in : f32 to i64
    linalg.yield %117 : i64
    } -> tensor<64x1xi64>
  %17 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<64x1xi64>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.subi %in, %c1_i64 : i64
    linalg.yield %117 : i64
    } -> tensor<64x1xi64>
  %18 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<64x1xi64>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.addi %in, %c1_i64 : i64
    linalg.yield %117 : i64
    } -> tensor<64x1xi64>
  %19 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<64x1xi64>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.addi %in, %c2_i64 : i64
    linalg.yield %117 : i64
    } -> tensor<64x1xi64>
  %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : tensor<64xi64>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.subi %in, %c1_i64 : i64
    linalg.yield %117 : i64
    } -> tensor<64xi64>
  %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : tensor<64xi64>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.addi %in, %c1_i64 : i64
    linalg.yield %117 : i64
    } -> tensor<64xi64>
  %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : tensor<64xi64>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.addi %in, %c2_i64 : i64
    linalg.yield %117 : i64
    } -> tensor<64xi64>
  %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %cst_2, %in : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.addf %in, %cst_2 : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<64xf32>) outs(%2 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %cst_0, %in : f32
    linalg.yield %117 : f32
    } -> tensor<64xf32>
  %27 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%concat_13 : tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.mulf %in, %cst_3 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %in, %cst_4 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%28, %concat_13 : tensor<2x64xf32>, tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%29 : tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.addf %in, %cst_5 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %31 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %concat_13 : tensor<2x64xf32>, tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %32 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %in, %cst_6 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %33 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%concat : tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.mulf %in, %cst_7 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %34 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %in, %cst_8 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %35 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%34, %concat : tensor<2x64xf32>, tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %36 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%35, %concat : tensor<2x64xf32>, tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %37 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%36 : tensor<2x64xf32>) outs(%26 : tensor<2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.addf %in, %cst_2 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64xf32>
  %38 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<64x1xf32>) outs(%8 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %cst_2, %in : f32
    linalg.yield %117 : f32
    } -> tensor<64x1xf32>
  %39 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<64x1xf32>) outs(%8 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.addf %in, %cst_2 : f32
    linalg.yield %117 : f32
    } -> tensor<64x1xf32>
  %40 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<64x1xf32>) outs(%8 : tensor<64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %cst_0, %in : f32
    linalg.yield %117 : f32
    } -> tensor<64x1xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%concat_25 : tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.mulf %in, %cst_3 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %43 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42 : tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %in, %cst_4 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %44 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %concat_25 : tensor<2x64x1xf32>, tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %45 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44 : tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.addf %in, %cst_5 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %46 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %concat_25 : tensor<2x64x1xf32>, tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %47 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46 : tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %in, %cst_6 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %48 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%concat_22 : tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.mulf %in, %cst_7 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %49 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48 : tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.subf %in, %cst_8 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %50 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %concat_22 : tensor<2x64x1xf32>, tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %51 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %concat_22 : tensor<2x64x1xf32>, tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %52 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51 : tensor<2x64x1xf32>) outs(%41 : tensor<2x64x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %117 = arith.addf %in, %cst_2 : f32
    linalg.yield %117 : f32
    } -> tensor<2x64x1xf32>
  %53 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<64x1xi64>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64x1xi64>
  %54 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%20 : tensor<64xi64>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64xi64>
  %56 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%55 : tensor<128xi64>) {
    ^bb0(%out: i64):
    %117 = linalg.index 0 : index
    %118 = arith.index_cast %117 : index to i64
    linalg.yield %118 : i64
    } -> tensor<128xi64>
  %58 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%57 : tensor<1xi64>) {
    ^bb0(%out: i64):
    linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
  %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %53, %54 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %61 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : tensor<64xi64>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64xi64>
  %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %53, %61 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %63 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%21 : tensor<64xi64>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64xi64>
  %64 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %53, %63 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %65 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%22 : tensor<64xi64>) outs(%0 : tensor<64xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64xi64>
  %66 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %53, %65 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %67 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %collapsed : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %68 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %collapsed_18 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %69 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %68 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %70 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %collapsed_19 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %71 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %70 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %72 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %collapsed_15 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %73 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %72 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %74 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<64x1xi64>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64x1xi64>
  %75 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %74, %54 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %76 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %74, %61 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %77 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %74, %63 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %78 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %74, %65 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %79 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %collapsed : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %80 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76, %collapsed_18 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %81 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79, %80 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %82 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%77, %collapsed_19 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %83 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%81, %82 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %84 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78, %collapsed_15 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %85 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%83, %84 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %86 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<64x1xi64>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64x1xi64>
  %87 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %86, %54 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %88 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %86, %61 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %89 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %86, %63 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %90 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %86, %65 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %91 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%87, %collapsed : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %92 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %collapsed_18 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %93 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %92 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %94 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %collapsed_19 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %95 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %94 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %96 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90, %collapsed_15 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %97 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %96 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %98 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<64x1xi64>) outs(%15 : tensor<64x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.select %117, %c0_i64, %in : i64
    %119 = arith.cmpi sge, %118, %c127_i64 : i64
    %120 = arith.select %119, %c127_i64, %118 : i64
    linalg.yield %120 : i64
    } -> tensor<64x1xi64>
  %99 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %98, %54 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %100 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %98, %61 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %101 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %98, %63 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %102 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35, %expanded_34, %98, %65 : tensor<1x1x1x1xi64>, tensor<128x1x1xi64>, tensor<64x1xi64>, tensor<64xi64>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: i64, %in_36: i64, %in_37: i64, %in_38: i64, %out: f32):
    %117 = arith.cmpi slt, %in, %c0_i64 : i64
    %118 = arith.addi %in, %c1_i64 : i64
    %119 = arith.select %117, %118, %in : i64
    %120 = arith.index_cast %119 : i64 to index
    %121 = arith.cmpi slt, %in_36, %c0_i64 : i64
    %122 = arith.addi %in_36, %c128_i64 : i64
    %123 = arith.select %121, %122, %in_36 : i64
    %124 = arith.index_cast %123 : i64 to index
    %125 = arith.cmpi slt, %in_37, %c0_i64 : i64
    %126 = arith.addi %in_37, %c128_i64 : i64
    %127 = arith.select %125, %126, %in_37 : i64
    %128 = arith.index_cast %127 : i64 to index
    %129 = arith.cmpi slt, %in_38, %c0_i64 : i64
    %130 = arith.addi %in_38, %c128_i64 : i64
    %131 = arith.select %129, %130, %in_38 : i64
    %132 = arith.index_cast %131 : i64 to index
    %extracted = tensor.extract %arg0[%120, %124, %128, %132] : tensor<1x128x128x128xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x128x64x64xf32>
  %103 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %collapsed : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %104 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%100, %collapsed_18 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %105 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %104 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %106 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %collapsed_19 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %107 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %106 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %108 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %collapsed_15 : tensor<1x128x64x64xf32>, tensor<64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %109 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %108 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %110 = linalg.generic {indexing_maps = [#map7, #map5, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %collapsed_28 : tensor<1x128x64x64xf32>, tensor<64x1xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %111 = linalg.generic {indexing_maps = [#map7, #map5, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%85, %collapsed_32 : tensor<1x128x64x64xf32>, tensor<64x1xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %112 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %111 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %113 = linalg.generic {indexing_maps = [#map7, #map5, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %collapsed_33 : tensor<1x128x64x64xf32>, tensor<64x1xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %114 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %113 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %115 = linalg.generic {indexing_maps = [#map7, #map5, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %collapsed_29 : tensor<1x128x64x64xf32>, tensor<64x1xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.mulf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
  %116 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %115 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) outs(%59 : tensor<1x128x64x64xf32>) {
    ^bb0(%in: f32, %in_36: f32, %out: f32):
    %117 = arith.addf %in, %in_36 : f32
    linalg.yield %117 : f32
    } -> tensor<1x128x64x64xf32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten._to_copy.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 128 (parallel)
    d3: 128 (parallel)
  indexing_maps:
    args_0 (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    _to_copy (1, 128, 128, 128) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    arange (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    arange (64,) operand: affine_map<(d0) -> (d0)>
    _to_copy_1 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    arange_1 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    arange_1 (64,) operand: affine_map<(d0) -> (d0)>
    _to_copy_2 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_2 (64,) operand: affine_map<(d0) -> (d0)>
    add (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.mul.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add (64,) operand: affine_map<(d0) -> (d0)>
    mul (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    mul (64,) operand: affine_map<(d0) -> (d0)>
    sub (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_1 (64,) operand: affine_map<(d0) -> (d0)>
    add_1 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.mul.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_1 (64,) operand: affine_map<(d0) -> (d0)>
    mul_1 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    mul_1 (64,) operand: affine_map<(d0) -> (d0)>
    sub_1 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    sub_1 (64,) operand: affine_map<(d0, d1) -> (d1)>
    unsqueeze (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.floor.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    sub (64,) operand: affine_map<(d0) -> (d0)>
    floor (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.floor.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    unsqueeze (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    floor_1 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    unsqueeze (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    floor_1 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    sub_2 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    sub_2 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    sub (64,) operand: affine_map<(d0) -> (d0)>
    floor (64,) operand: affine_map<(d0) -> (d0)>
    sub_3 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    sub_3 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_1 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    floor (64,) operand: affine_map<(d0) -> (d0)>
    _to_copy_3 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    floor_1 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    _to_copy_4 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    sub_4 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    add_2 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    add_3 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_3 (64,) operand: affine_map<(d0) -> (d0)>
    sub_5 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_3 (64,) operand: affine_map<(d0) -> (d0)>
    add_4 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_3 (64,) operand: affine_map<(d0) -> (d0)>
    add_5 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    clamp_1 (64,) operand: affine_map<(d0) -> (d0)>
    sub_6 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.cat.default
  iterator_types = ['parallel']
  loops:
    d0: 128 (parallel)
  indexing_maps:
    clamp_1 (64,) operand: affine_map<(d0) -> (d0)>
    sub_6 (64,) operand: affine_map<(d0) -> (d0)>
    cat (128,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.view.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    cat (128,) operand: affine_map<(d0, d1) -> (d1)>
    view (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    clamp_1 (64,) operand: affine_map<(d0) -> (d0)>
    add_6 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    clamp_1 (64,) operand: affine_map<(d0) -> (d0)>
    sub_7 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.cat.default
  iterator_types = ['parallel']
  loops:
    d0: 128 (parallel)
  indexing_maps:
    add_6 (64,) operand: affine_map<(d0) -> (d0)>
    sub_7 (64,) operand: affine_map<(d0) -> (d0)>
    cat_1 (128,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.view.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    cat_1 (128,) operand: affine_map<(d0, d1) -> (d1)>
    view_1 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    view_1 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_2 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    mul_2 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub_8 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    sub_8 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    view_1 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_3 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    mul_3 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    add_7 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    add_7 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    view_1 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_4 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    mul_4 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub_9 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    view (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_5 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    mul_5 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub_10 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    sub_10 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    view (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_6 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    mul_6 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    view (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_7 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    mul_7 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    add_8 (2, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    sub_9 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    slice_1 (1, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    sub_9 (2, 64) operand: affine_map<(d0, d1) -> (d0 + 1, d1)>
    slice_2 (1, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.squeeze.dims
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    slice_1 (1, 64) operand: affine_map<(d0) -> (0, d0)>
    squeeze (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.squeeze.dims
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    slice_2 (1, 64) operand: affine_map<(d0) -> (0, d0)>
    squeeze_1 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    add_8 (2, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    slice_3 (1, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    add_8 (2, 64) operand: affine_map<(d0, d1) -> (d0 + 1, d1)>
    slice_4 (1, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.squeeze.dims
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    slice_3 (1, 64) operand: affine_map<(d0) -> (0, d0)>
    squeeze_2 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.squeeze.dims
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    slice_4 (1, 64) operand: affine_map<(d0) -> (0, d0)>
    squeeze_3 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    clamp (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    sub_11 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.cat.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 128 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    clamp (64, 1) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub_11 (64, 1) operand: affine_map<(d0, d1) -> (d0, d1)>
    cat_2 (128, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.view.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    cat_2 (128, 1) operand: affine_map<(d0, d1, d2) -> (d1, 0)>
    view_2 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.add.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    clamp (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    add_9 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    clamp (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    sub_12 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.cat.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 128 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_9 (64, 1) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub_12 (64, 1) operand: affine_map<(d0, d1) -> (d0, d1)>
    cat_3 (128, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.view.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    cat_3 (128, 1) operand: affine_map<(d0, d1, d2) -> (d1, 0)>
    view_3 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    view_3 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    mul_8 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    mul_8 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    sub_13 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    sub_13 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    view_3 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    mul_9 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    mul_9 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    add_10 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    add_10 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    view_3 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    mul_10 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    mul_10 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    sub_14 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    view_2 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    mul_11 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    mul_11 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    sub_15 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    sub_15 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    view_2 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    mul_12 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    mul_12 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    view_2 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    mul_13 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 2 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    mul_13 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    add_11 (2, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    sub_14 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    slice_5 (1, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    sub_14 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0 + 1, d1, d2)>
    slice_6 (1, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.squeeze.dims
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    slice_5 (1, 64, 1) operand: affine_map<(d0, d1) -> (0, d0, 0)>
    squeeze_4 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.squeeze.dims
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    slice_6 (1, 64, 1) operand: affine_map<(d0, d1) -> (0, d0, 0)>
    squeeze_5 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    add_11 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    slice_7 (1, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.slice.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 64 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    add_11 (2, 64, 1) operand: affine_map<(d0, d1, d2) -> (d0 + 1, d1, d2)>
    slice_8 (1, 64, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.squeeze.dims
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    slice_7 (1, 64, 1) operand: affine_map<(d0, d1) -> (0, d0, 0)>
    squeeze_6 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.squeeze.dims
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    slice_8 (1, 64, 1) operand: affine_map<(d0, d1) -> (0, d0, 0)>
    squeeze_7 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    sub_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_2 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    sub_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_3 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_2 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_3 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    sub_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_4 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_3 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_5 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_4 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_5 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_1 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    sub_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_6 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_4 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_7 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_6 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_7 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_2 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    sub_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_8 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_9 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_8 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_9 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_3 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_14 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_1 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_2 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_15 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    mul_14 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_15 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_12 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_2 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_3 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_16 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_12 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_16 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_13 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_3 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_1 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_17 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_13 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_17 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_14 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_10 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    sub_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_11 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_10 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_11 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_4 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_12 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_3 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_13 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_12 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_13 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_5 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_14 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_4 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_15 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_14 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_15 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_6 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_4 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_16 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_17 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_16 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_17 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_7 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_4 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_18 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_5 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_2 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_19 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    mul_18 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_19 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_15 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_6 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_3 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_20 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_15 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_20 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_16 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_7 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_1 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_21 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_16 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_21 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_17 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_2 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_18 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    sub_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_19 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_18 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_19 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_8 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_2 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_20 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_3 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_21 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_20 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_21 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_9 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_2 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_22 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_4 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_23 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_22 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_23 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_10 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_2 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_24 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_25 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_24 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_25 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_11 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_8 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_22 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_9 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_2 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_23 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    mul_22 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_23 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_18 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_10 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_3 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_24 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_18 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_24 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_19 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_11 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_1 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_25 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_19 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_25 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_20 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_3 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_26 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    sub_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_27 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_26 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_27 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_12 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_3 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_28 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    _to_copy_3 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_29 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_28 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_29 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_13 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_3 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_30 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_4 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_31 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_30 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_31 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_14 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 64 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_3 (64, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_32 (64, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 64 (parallel)
  indexing_maps:
    add_5 (64,) operand: affine_map<(d0) -> (d0)>
    clamp_33 (64,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    _to_copy (1, 128, 128, 128) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_32 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_33 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_15 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_12 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_26 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_13 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_2 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_27 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    mul_26 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_27 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_21 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_14 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_3 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_28 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_21 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_28 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_22 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    index_15 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_1 (64,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_29 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_22 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_29 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_23 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_14 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_4 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    mul_30 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_17 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_6 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    mul_31 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    mul_30 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_31 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_24 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_20 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_7 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    mul_32 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_24 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_32 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_25 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_23 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    squeeze_5 (64, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    mul_33 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_25 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_33 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_26 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten._to_copy.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 128 (parallel)
    d2: 64 (parallel)
    d3: 64 (parallel)
  indexing_maps:
    add_26 (1, 128, 64, 64) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    _to_copy_5 (1, 128, 64, 64) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3

Working set: 10485760 bytes -> tier T3.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 10485760 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 10485760 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
  vtcm: the staged tiles need a software-managed scratchpad
        working set 10485760 B also exceeds VTCM 8388608 B, so it must be streamed in tiles -- double-buffered

Provenance (this governs whether the result is usable at all):
  - Derive this kernel from the reference, the schedule, and the vendor's
    intrinsic headers ONLY.
  - NOTHING from this repository may appear in the translation unit -- no
    `harness_common.h`, no `hmx_helpers.h`, nothing else under
    hexbench/env/harness/. Those are R&D material. The directory is not on the
    compile's include path, so such an include is a build failure, and a COPIED
    body (the VTCM base, the alignment attribute, a crouton offset, a tolerance
    compare) is rejected before compiling. Hardware facts are yours to derive
    under your own names, with a comment saying where you got them.
  - Do not copy from, or pattern it on, any existing solution to this or a
    similar kernel -- including any private reference corpus and including other
    kernels generated alongside this one.
  - This pipeline's claim is that its corpus comes from public sources: an op
    list from the operator registry, sizes from the memory hierarchy, mechanisms
    computed from those sizes. A kernel copied from an existing solution breaks
    that claim no matter how well it performs, and cannot be used.

Rules:
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_args_0, float *out0)
    The "extern \"C\"" is required; without it the symbol mangles and the
    harness fails to link.
  - It is compiled as C++17 and run on the simulator against golden vectors.
    Producing the wrong numbers fails, however fast it is.
  - Do not change the signature, the parameter order, or the memory layout.
    Parameter order follows the traced graph's placeholders.
  - Include what you use. The vendor headers are available and are the only ones
    that resolve: <hexagon_types.h> (which also carries the user-DMA descriptor
    types), <hexagon_protos.h>, <hvx_hexagon_protos.h>, <hmx_hexagon_protos.h>,
    plus the C++ standard library. No repo-local header is on the include path.
  - The harness enables the HMX context before calling you, so HMX code needs no
    SSR setup (doing it anyway is harmless -- it is idempotent).
  - If you use the DMA engine, declare every descriptor
    `static hexagon_udma_descriptor_typeN_t d[...] __attribute__((aligned(64)));`.
    A descriptor is read by an EXTERNAL engine and does not tolerate the alignment
    a stack frame happens to give a 16- or 32-byte struct. Measured: the same
    kernel with function-local descriptors failed 2,979 of 3,211,264 elements and
    with static aligned ones passed 0 of 3,211,264, pipeline untouched. A second
    kernel crashed 0x28 the same way.
  - Also: a second `dmstart` may only be outstanding after the program's FIRST
    transfer has completed. Prime the engine with one `dmstart`/`dmwait` in a
    prologue and then overlap freely, or chain descriptors through `next` and issue
    ONE `dmstart` per tile. Two `dmstart`s as the engine's first act fault 0x28.
  - Return only the C++ translation unit, no prose and no markdown fence.
