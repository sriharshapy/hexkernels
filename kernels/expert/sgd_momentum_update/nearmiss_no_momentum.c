/* NEAR-MISS: ignores the momentum buffer v entirely.
 * Updates w with a plain SGD step: w[i] -= lr*grad[i].
 * v[] is left unchanged (or written as grad[i] without mu accumulation).
 * Compiles fine; fails because v is never accumulated and w update skips history.
 */
void candidate_kernel(float *w, float *v, const float *grad, int n,
                      float lr, float mu) {
    for (int i = 0; i < n; i++) {
        /* WRONG: ignores momentum; v not properly accumulated */
        v[i] = grad[i];            /* loses mu*v carry */
        w[i] = w[i] - lr * grad[i]; /* should be lr*v[i] after update */
    }
}
