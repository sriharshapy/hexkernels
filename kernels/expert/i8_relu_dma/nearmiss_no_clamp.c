#include "kernel_api.h"
void candidate_kernel(const int8_t *x, int8_t *out, int n) { for (int i=0;i<n;i++) out[i]=x[i]; }
