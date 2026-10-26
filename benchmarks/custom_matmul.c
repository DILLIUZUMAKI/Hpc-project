/* Custom optimized serial replacement for Dense Matrix Multiplication */
#include "common.h"

#define BLOCK_SIZE 32

int main(int argc, char **argv) {
    Config cfg = config(argc, argv, 1, 1024);
    int n = cfg.n;
    size_t nn = (size_t)n * n;

    double *a = allocate(nn, sizeof(double));
    double *b = allocate(nn, sizeof(double));
    double *c = allocate(nn, sizeof(double));

    for (size_t i = 0; i < nn; i++) {
        a[i] = initial((uint32_t)(2 * i), cfg.seed);
        b[i] = initial((uint32_t)(2 * i + 1), cfg.seed);
    }

    double t = now();

    /* Cache-blocked / tiled matrix multiplication */
    for (int sj = 0; sj < n; sj += BLOCK_SIZE) {
        int ej = (sj + BLOCK_SIZE < n) ? sj + BLOCK_SIZE : n;
        for (int sk = 0; sk < n; sk += BLOCK_SIZE) {
            int ek = (sk + BLOCK_SIZE < n) ? sk + BLOCK_SIZE : n;
            for (int i = 0; i < n; i++) {
                size_t in = (size_t)i * n;
                for (int k = sk; k < ek; k++) {
                    double aik = a[in + k];
                    size_t kn = (size_t)k * n;
                    for (int j = sj; j < ej; j++) {
                        c[in + j] += aik * b[kn + j];
                    }
                }
            }
        }
    }

    double elapsed = now() - t;
    emit(c, nn, cfg.full, elapsed, 0);

    free(a);
    free(b);
    free(c);
    return 0;
}
