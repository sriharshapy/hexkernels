/* Correct scalar baseline: in-place AXPY across T tensors of length L.
 * Semantics: y[t*L+i] = alpha * x[t*L+i] + y[t*L+i] */
void candidate_kernel(const float *x, float *y, int T, int L, float alpha) {
    for (int t = 0; t < T; t++) {
        int base = t * L;
        for (int i = 0; i < L; i++)
            y[base + i] = alpha * x[base + i] + y[base + i];
    }
}
