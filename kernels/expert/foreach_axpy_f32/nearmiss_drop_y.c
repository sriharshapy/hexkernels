/* Near-miss: overwrites y = alpha*x, dropping the in-place accumulation (+y).
 * Fails whenever y_orig[t*L+i] != 0, which is common in seeded data. */
void candidate_kernel(const float *x, float *y, int T, int L, float alpha) {
    for (int t = 0; t < T; t++) {
        int base = t * L;
        for (int i = 0; i < L; i++)
            y[base + i] = alpha * x[base + i];  /* WRONG: drops the + y[base+i] term */
    }
}
