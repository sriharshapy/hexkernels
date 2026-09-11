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
