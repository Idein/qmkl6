#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

#include "cblasdefs.h"

static int test_sscal_single(const size_t n) {
  float *x = (decltype(x))mkl_malloc(sizeof(*x) * n, 64);

  const float a = std::atan(1.f) * 4;

  for (size_t i = 0; i < n; ++i) x[i] = i;

  const double start = dsecnd();
  cblas_sscal(n, a, x, 1);
  const double end = dsecnd();
  printf("%zu bytes, %f sec, %f MB/s\n", sizeof(*x) * n, end - start,
         sizeof(*x) * n / (end - start) * 1e-6);

  float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
  float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
  for (size_t i = 0; i < n; ++i) {
    const float expected = a * i;
    const float err_abs = std::abs(x[i] - expected);
    const float err_rel = std::abs(err_abs / expected);
    err_abs_min = std::min(err_abs_min, err_abs);
    err_abs_max = std::max(err_abs_max, err_abs);
    err_rel_min = std::min(err_rel_min, err_rel);
    err_rel_max = std::max(err_rel_max, err_rel);
  }
  printf("Minimum/maximum absolute errors: %e, %e\n", err_abs_min, err_abs_max);
  printf("Minimum/maximum relative errors: %e, %e\n", err_rel_min, err_rel_max);

  if (err_rel_max > 1e-3f) {
    std::cerr << "error: Relative error is too large" << std::endl;
    return 1;
  }

  mkl_free(x);
  return 0;
}

static int test_sscal_random(void) {
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dist_n(1, 1 << 18);
  std::uniform_int_distribution<unsigned> dist_inc(1, 64);
  std::uniform_real_distribution<float> dist_value;

  for (unsigned i = 0; i < 20; ++i) {
    const unsigned n = dist_n(gen);
    const unsigned incx = dist_inc(gen);
    const float a = dist_value(gen);
    printf("Testing n = %u, incx = %u, a = %f\n", n, incx, a);

    float *x = (decltype(x))mkl_malloc(sizeof(*x) * n * incx + incx - 1, 64);
    float *x_orig = (decltype(x_orig))malloc(sizeof(*x_orig) * n);

    if (x_orig == NULL) {
      std::cerr << "error: Failed to allocate reference vector" << std::endl;
      return 1;
    }

    for (size_t j = 0, k = 0; j < n; ++j, k += incx) {
      x[k] = dist_value(gen);
      x_orig[j] = a * x[k];
    }

    cblas_sscal(n, a, x, incx);

    float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
    float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
    for (size_t j = 0, k = 0; j < n; ++j, k += incx) {
      const float err_abs = std::abs(x[k] - x_orig[j]);
      const float err_rel = std::abs(err_abs / x_orig[j]);
      err_abs_min = std::min(err_abs_min, err_abs);
      err_abs_max = std::max(err_abs_max, err_abs);
      err_rel_min = std::min(err_rel_min, err_rel);
      err_rel_max = std::max(err_rel_max, err_rel);
    }
    printf("Minimum/maximum absolute errors: %e, %e\n", err_abs_min,
           err_abs_max);
    printf("Minimum/maximum relative errors: %e, %e\n", err_rel_min,
           err_rel_max);

    if (err_rel_max > 1e-3f) {
      std::cerr << "error: The maximum relative error is too large"
                << std::endl;
      return 1;
    }

    mkl_free(x);
    free(x_orig);
  }

  return 0;
}

int main(void) {
  setlinebuf(stdout);

  int ret;

  ret = test_sscal_single(1 << 24);
  if (ret) return ret;

  ret = test_sscal_random();
  if (ret) return ret;

  return 0;
}
