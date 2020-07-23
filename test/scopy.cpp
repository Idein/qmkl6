#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

#include <qmkl6.h>


static
int test_scopy_single(const size_t n)
{
    uint32_t *x = (uint32_t*) mkl_malloc(sizeof(*x) * n, 64),
            *y = (uint32_t*) mkl_calloc(n, sizeof(*y), 64);

    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += x[i] = i;
    printf("Sum (expected): %" PRIu64 "\n", sum);

    sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += y[i];
    printf("Sum (before execution): %" PRIu64 "\n", sum);

    if (sum != 0) {
        std::cerr << "error: Clear-allocated array is not zeroed" << std::endl;
        return 1;
    }

    const double start = dsecond();
    cblas_scopy(n, (const float*) x, 1, (float*) y, 1);
    const double end = dsecond();
    printf("%zu bytes, %f sec, %f MB/s\n", sizeof(uint32_t) * n, end - start,
            sizeof(uint32_t) * n / (end - start) * 1e-6);

    sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += y[i];
    printf("Sum (actual): %" PRIu64 "\n", sum);

    if (sum != (uint64_t) n * (n - 1) >> 1) {
        std::cerr << "error: The actual sum is different from expected"
                << std::endl;
        return 1;
    }

    mkl_free(x);
    mkl_free(y);
    return 0;
}

static
int test_scopy_random(void)
{
    std::default_random_engine gen;
    std::uniform_int_distribution <size_t> dist_n(1, 1 << 18);
    std::uniform_int_distribution <unsigned> dist_inc(1, 64);
    std::uniform_int_distribution <uint32_t> dist_value;

    for (unsigned i = 0; i < 20; ++i) {
        const unsigned n = dist_n(gen);
        const unsigned incx = dist_inc(gen), incy = dist_inc(gen);
        printf("Testing n = %u, incx = %u, incy = %u\n", n, incx, incy);

        uint32_t *x =
                (decltype(x)) mkl_malloc(sizeof(*x) * n * incx + incx - 1, 64),
                *y =
                (decltype(y)) mkl_malloc(sizeof(*y) * n * incy + incy - 1, 64);

        uint32_t sum_expected = 0, sum_actual = 0;
        for (size_t j = 0, k = 0; j < n; ++j, k += incx)
            sum_expected += x[k] = dist_value(gen);
        printf("Sum (expected): %" PRIu32 "\n", sum_expected);

        for (size_t j = 0; j < n; j += incy)
            sum_actual += y[j];
        printf("Sum (before execution): %" PRIu32 "\n", sum_actual);

        cblas_scopy(n, (const float*) x, incx, (float*) y, incy);

        sum_actual = 0;
        for (size_t j = 0, k = 0; j < n; ++j, k += incy)
            sum_actual += y[k];
        printf("Sum (after execution): %" PRIu32 "\n", sum_actual);

        if (sum_actual != sum_expected) {
            std::cerr << "error: The actual sum is different from expected"
                    << std::endl;
            return 1;
        }

        mkl_free(x);
        mkl_free(y);
    }

    return 0;
}

int main(void)
{
    setlinebuf(stdout);

    int ret;

    ret = test_scopy_single(1 << 24);
    if (ret)
        return ret;

    ret = test_scopy_random();
    if (ret)
        return ret;

    return 0;
}
