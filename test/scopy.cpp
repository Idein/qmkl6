#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

#include "cblasdefs.h"

static int test_memcpy_single(const size_t n) {
  uint32_t *x = new uint32_t[n], *y = new uint32_t[n];

  std::iota(x, x + n, 0);
  std::fill(y, y + n, 0);

  const double start = dsecond();
  memcpy(y, x, sizeof(uint32_t) * n);
  const double end = dsecond();

  printf("memcpy: %zu bytes, %f sec, %f MB/s\n", sizeof(uint32_t) * n,
         end - start, sizeof(uint32_t) * n / (end - start) * 1e-6);

  if (!std::all_of(y, y + n, [j = (uint32_t)0](const uint32_t v) mutable {
        return v == j++;
      })) {
    std::cerr << "error: Copy is not complete" << std::endl;
    return 1;
  }

  delete x;
  delete y;
  return 0;
}

static int test_scopy_single(const size_t n) {
  uint32_t *x = (uint32_t *)mkl_malloc(sizeof(*x) * n, 64),
           *y = (uint32_t *)mkl_calloc(n, sizeof(*y), 64);

  std::iota(x, x + n, 0);

  if (std::accumulate(y, y + n, 0) != 0) {
    std::cerr << "error: Clear-allocated array is not zeroed" << std::endl;
    return 1;
  }

  const double start = dsecond();
  cblas_scopy(n, (const float *)x, 1, (float *)y, 1);
  const double end = dsecond();
  printf("scopy: %zu bytes, %f sec, %f MB/s\n", sizeof(uint32_t) * n,
         end - start, sizeof(uint32_t) * n / (end - start) * 1e-6);

  if (!std::all_of(y, y + n, [j = (uint32_t)0](const uint32_t v) mutable {
        return v == j++;
      })) {
    std::cerr << "error: Copy is not complete" << std::endl;
    return 1;
  }

  mkl_free(x);
  mkl_free(y);
  return 0;
}

static int test_scopy_random(void) {
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dist_n(1, 1 << 18);
  std::uniform_int_distribution<unsigned> dist_inc(1, 64);
  std::uniform_int_distribution<uint32_t> dist_value;

  for (unsigned i = 0; i < 20; ++i) {
    const unsigned n = dist_n(gen);
    const unsigned incx = dist_inc(gen), incy = dist_inc(gen);
    printf("Testing n = %u, incx = %u, incy = %u\n", n, incx, incy);

    uint32_t *x = (decltype(x))mkl_malloc(sizeof(*x) * n * incx + incx - 1, 64),
             *y = (decltype(y))mkl_malloc(sizeof(*y) * n * incy + incy - 1, 64);

    uint32_t sum_expected = 0, sum_actual = 0;
    for (size_t j = 0, k = 0; j < n; ++j, k += incx)
      sum_expected += x[k] = dist_value(gen);
    printf("Sum (expected): %" PRIu32 "\n", sum_expected);

    for (size_t j = 0; j < n; j += incy) sum_actual += y[j];
    printf("Sum (before execution): %" PRIu32 "\n", sum_actual);

    cblas_scopy(n, (const float *)x, incx, (float *)y, incy);

    sum_actual = 0;
    for (size_t j = 0, k = 0; j < n; ++j, k += incy) sum_actual += y[k];
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

static int test_scopy_partial(const size_t n) {
  uint32_t *x = (uint32_t *)mkl_malloc(sizeof(*x) * n, 64),
           *y = (uint32_t *)mkl_malloc(sizeof(*y) * n, 64);
  std::iota(x, x + n, 0);

  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dist;

  for (unsigned i = 0; i < 20; ++i) {
    dist.param(decltype(dist)::param_type(1, n));
    const size_t len = dist(gen);

    dist.param(decltype(dist)::param_type(0, n - len));
    const size_t offset = dist(gen);

    printf("Testing n = %zu, len = %zu, offset = %zu\n", n, len, offset);

    std::fill(y, y + n, 0);
    cblas_scopy(len, (const float *)x + offset, 1, (float *)y, 1);

    if (!std::all_of(y, y + len, [j = offset](const uint32_t v) mutable {
          return v == j++;
        })) {
      std::cerr << "error: Short copy" << std::endl;
      return 1;
    }
    if (!std::all_of(y + len, y + n, [](const uint32_t v) { return v == 0; })) {
      std::cerr << "error: Extra copy" << std::endl;
      return 1;
    }
  }

  mkl_free(x);
  mkl_free(y);
  return 0;
}

int main(void) {
  setlinebuf(stdout);

  int ret;

  ret = test_memcpy_single(1 << 24);
  if (ret) return ret;

  ret = test_scopy_single(1 << 24);
  if (ret) return ret;

  ret = test_scopy_random();
  if (ret) return ret;

  ret = test_scopy_partial(1 << 24);
  if (ret) return ret;

  return 0;
}
