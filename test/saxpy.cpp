#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

#include "cblasdefs.h"


static
int test_saxpy_single(const size_t n)
{
    float *x = (float*) mkl_malloc(sizeof(*x) * n, 64),
            *y = (float*) mkl_malloc(sizeof(*y) * n, 64);
    float *y_orig = (float*) malloc(sizeof(*y_orig) * n);

    if (y_orig == NULL) {
        std::cerr << "error: Failed to allocate reference vector" << std::endl;
        return 1;
    }

    const float coef = 3.f;

    for (size_t i = 0; i < n; ++i) {
        x[i] = i;
        y[i] = -i;
        y_orig[i] = coef * x[i] + y[i];
    }

    const double start = dsecond();
    cblas_saxpy(n, coef, x, 1, y, 1);
    const double end = dsecond();

    printf("%zu elements, %f sec, %f Mflop/s\n", n, end - start,
            n / (end - start) * 1e-6);

    float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
    float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
    for (size_t i = 0; i < n; ++i) {
        const float err_abs = std::abs(y[i] - y_orig[i]);
        const float err_rel = std::abs(err_abs / y_orig[i]);
        err_abs_min = std::min(err_abs_min, err_abs);
        err_abs_max = std::max(err_abs_max, err_abs);
        err_rel_min = std::min(err_rel_min, err_rel);
        err_rel_max = std::max(err_rel_max, err_rel);
    }
    printf("Minimum/maximum absolute errors: %e, %e\n",
            err_abs_min, err_abs_max);
    printf("Minimum/maximum relative errors: %e, %e\n",
            err_rel_min, err_rel_max);

    if (err_abs_max != 0.f || err_rel_max != 0.f) {
        std::cerr << "error: The results contain too large errors" << std::endl;
        return 1;
    }

    mkl_free(x);
    mkl_free(y);
    free(y_orig);
    return 0;
}

static
int test_saxpy_random(void)
{
    std::default_random_engine gen;
    std::uniform_int_distribution <size_t> dist_n(1, 1 << 18);
    std::uniform_int_distribution <unsigned> dist_inc(1, 64);
    std::uniform_real_distribution <float> dist_value;

    for (unsigned i = 0; i < 20; ++i) {
        const unsigned n = dist_n(gen);
        const unsigned incx = dist_inc(gen), incy = dist_inc(gen);
        const float coef = dist_value(gen);
        printf("Testing n = %u, incx = %u, incy = %u, coef = %e\n",
                n, incx, incy, coef);

        float *x =
                (decltype(x)) mkl_malloc(sizeof(*x) * n * incx + incx - 1, 64),
                *y =
                (decltype(y)) mkl_malloc(sizeof(*y) * n * incy + incy - 1, 64);
        float *y_orig = (float*) malloc(sizeof(*y_orig) * n);

        if (y_orig == NULL) {
            std::cerr << "error: Failed to allocate reference vector"
                    << std::endl;
            return 1;
        }

        for (size_t j = 0, k = 0, l = 0; j < n; ++j, k += incx, l += incy) {
            x[k] = dist_value(gen);
            y[l] = dist_value(gen);
            y_orig[j] = coef * x[k] + y[l];
        }

        cblas_saxpy(n, coef, x, incx, y, incy);

        float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
        float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
        for (size_t j = 0, k = 0; j < n; ++j, k += incy) {
            const float err_abs = std::abs(y[k] - y_orig[j]);
            const float err_rel = std::abs(err_abs / y_orig[j]);
            err_abs_min = std::min(err_abs_min, err_abs);
            err_abs_max = std::max(err_abs_max, err_abs);
            err_rel_min = std::min(err_rel_min, err_rel);
            err_rel_max = std::max(err_rel_max, err_rel);
        }
        printf("Minimum/maximum absolute errors: %e, %e\n",
                err_abs_min, err_abs_max);
        printf("Minimum/maximum relative errors: %e, %e\n",
                err_rel_min, err_rel_max);

        if (err_rel_max != 0.f) {
            std::cerr << "error: The maximum relative error is too large"
                    << std::endl;
            return 1;
        }

        mkl_free(x);
        mkl_free(y);
        free(y_orig);
    }

    return 0;
}

int main(void)
{
    setlinebuf(stdout);

    int ret;

    ret = test_saxpy_single(1 << 24);
    if (ret)
        return ret;

    ret = test_saxpy_random();
    if (ret)
        return ret;

    return 0;
}
