#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

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

int main(void)
{
    int ret;

    ret = test_scopy_single(1 << 24);
    if (ret)
        return ret;

    return 0;
}
