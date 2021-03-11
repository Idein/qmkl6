#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

#include "cblasdefs.h"

static int test_sasum_single(const size_t n) {
  float *x = (float *)mkl_malloc(sizeof(*x) * n, 64);

  float asum_exp = 0.f;
  for (size_t i = 0; i < n; ++i) {
    x[i] = (i & 1) ? -(float)i : i;
    asum_exp += i;
  }

  printf("Absolute sum (expected): %e\n", asum_exp);

  const double start = dsecnd();
  const float asum_act = cblas_sasum(n, x, 1);
  const double end = dsecnd();

  printf("Absolute sum (actual): %e\n", asum_act);

  printf("%zu elements, %f sec, %f Mflop/s\n", n, end - start,
         n / (end - start) * 1e-6);

  const float err_abs = std::abs(asum_act - asum_exp);
  const float err_rel = std::abs(err_abs / asum_exp);

  printf("Absolute/relative errors: %e, %e\n", err_abs, err_rel);

  if (err_rel > 1e-5 * asum_exp) {
    std::cerr << "error: The relative error is too large" << std::endl;
    return 1;
  }

  mkl_free(x);
  return 0;
}

static int test_sasum_random(void) {
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dist_n(1, 1 << 18);
  std::uniform_int_distribution<unsigned> dist_inc(1, 64);
  std::uniform_real_distribution<float> dist_value(-1.f, 1.f);

  for (unsigned i = 0; i < 20; ++i) {
    const unsigned n = dist_n(gen);
    const unsigned incx = dist_inc(gen);
    printf("Testing n = %u, incx = %u\n", n, incx);

    float *x = (decltype(x))mkl_malloc(sizeof(*x) * n * incx + incx - 1, 64);

    float asum_exp = 0.f;
    for (size_t j = 0, k = 0; j < n; ++j, k += incx) {
      x[k] = dist_value(gen);
      asum_exp += std::abs(x[k]);
    }

    const float asum_act = cblas_sasum(n, x, incx);
    printf("Absolute sum expected/actual: %e, %e\n", asum_exp, asum_act);

    const float err_abs = std::abs(asum_act - asum_exp);
    const float err_rel = std::abs(err_abs / asum_exp);

    printf("Absolute/relative errors: %e, %e\n", err_abs, err_rel);

    if (err_rel > 1e-5 * asum_exp) {
      std::cerr << "error: The relative error is too large" << std::endl;
      return 1;
    }

    mkl_free(x);
  }

  return 0;
}

int main(void) {
  setlinebuf(stdout);

  int ret;

  ret = test_sasum_single(1 << 24);
  if (ret) return ret;

  ret = test_sasum_random();
  if (ret) return ret;

  return 0;
}
