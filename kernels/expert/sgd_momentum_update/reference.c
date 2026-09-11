/* Correct scalar baseline: SGD with momentum in-place update.
 * v[i] = mu*v[i] + grad[i]
 * w[i] = w[i] - lr*v[i]
 */
void candidate_kernel(float *w, float *v, const float *grad, int n,
                      float lr, float mu) {
    for (int i = 0; i < n; i++) {
        v[i] = mu * v[i] + grad[i];
        w[i] = w[i] - lr * v[i];
    }
}
