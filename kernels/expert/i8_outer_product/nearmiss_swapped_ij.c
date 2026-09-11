/* Near-miss: swaps i and j indices — writes a[j]*b[i] instead of a[i]*b[j].
   Fails whenever a and b differ (i.e. always on random inputs). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int32_t *C,
                      int M, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i*N+j] = (int32_t)a[j] * (int32_t)b[i];  /* swapped: a[j]*b[i] instead of a[i]*b[j] */
}
