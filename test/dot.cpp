#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

#include "cblasdefs.h"

static int test_sdot_single(const size_t n) {
  float *x = (float *)mkl_malloc(sizeof(*x) * n, 64),
        *y = (float *)mkl_malloc(sizeof(*y) * n, 64);

  float res_exp = 0.f;
  for (size_t i = 0; i < n; ++i) {
    x[i] = 1.f;
    y[i] = 2.f;
    res_exp += x[i] * y[i];
  }
  printf("Result (expected): %e\n", res_exp);

  const double start = dsecnd();
  const float res_act = cblas_sdot(n, x, 1, y, 1);
  const double end = dsecnd();

  const float err_abs = std::abs(res_exp - res_act);
  const float err_rel = std::abs(err_abs / res_exp);

  printf("Result (actual): %e\n", res_act);
  printf("Absolute error: %e\n", err_abs);
  printf("Relative error: %e\n", err_rel);

  printf("%zu elements, %f sec, %f Mflop/s\n", n, end - start,
         n / (end - start) * 1e-6);

  if (err_rel > 1e-3f) {
    std::cerr << "error: Relative error is too large" << std::endl;
    return 1;
  }

  mkl_free(x);
  mkl_free(y);
  return 0;
}

static int test_sdot_random(void) {
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dist_n(1, 1 << 18);
  std::uniform_int_distribution<unsigned> dist_inc(1, 64);
  std::uniform_real_distribution<float> dist_value;

  for (unsigned i = 0; i < 20; ++i) {
    const unsigned n = dist_n(gen);
    const unsigned incx = dist_inc(gen), incy = dist_inc(gen);
    printf("Testing n = %u, incx = %u, incy = %u\n", n, incx, incy);

    float *x = (decltype(x))mkl_malloc(sizeof(*x) * n * incx + incx - 1, 64),
          *y = (decltype(y))mkl_malloc(sizeof(*y) * n * incy + incy - 1, 64);

    float res_exp = 0.f;
    for (size_t j = 0, k = 0, l = 0; j < n; ++j, k += incx, l += incy)
      res_exp += (x[k] = dist_value(gen)) * (y[l] = dist_value(gen));
    printf("Result (expected): %f\n", res_exp);

    const float res_act = cblas_sdot(n, x, incx, y, incy);
    printf("Result (actual): %f\n", res_act);

    const float err_abs = std::abs(res_exp - res_act);
    const float err_rel = std::abs(err_abs / res_act);
    printf("Absolute/relative errors: %e, %e\n", err_abs, err_rel);

    if (err_abs > 1e-8f + 1e-3f * std::abs(res_exp)) {
      std::cerr << "error: The absolute error is too large" << std::endl;
      return 1;
    }

    mkl_free(x);
    mkl_free(y);
  }

  return 0;
}

int main(void) {
  setlinebuf(stdout);

  int ret;

  ret = test_sdot_single(1 << 24);
  if (ret) return ret;

  ret = test_sdot_random();
  if (ret) return ret;

  return 0;
}
