/* NEAR-MISS B: identity -- no clamping at all.
 * Fails for x < 0 (should be 0) and x > 6 (should be 6). */
void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i];  /* no clamp: passes all values unchanged */
    }
}
