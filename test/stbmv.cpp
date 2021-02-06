#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>

#include "cblasdefs.h"

static void naive_stbmv([[maybe_unused]] const CBLAS_LAYOUT layout,
                        [[maybe_unused]] const CBLAS_UPLO uplo,
                        [[maybe_unused]] const CBLAS_TRANSPOSE trans,
                        [[maybe_unused]] const CBLAS_DIAG diag, const int n,
                        const int k, const float *const a, const int lda,
                        float *const x, const int incx) {
  if (k != 0) {
    std::cerr << "error: We only support diagonal matrices for now"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

#pragma omp parallel for default(none) firstprivate(n, a, lda, x, incx) \
    schedule(guided)
  for (int i = 0; i < n; ++i) x[incx * i] *= a[lda * i];
}

template <class Generator>
static int test_stbmv_diag_single(const int n, const int lda, const int incx,
                                  Generator &gen) {
  float *a, *x0, *x1;
  std::uniform_real_distribution<float> dist;

  printf("n = %5d, lda = %5d, incx = %d: ", n, lda, incx);

  a = (float *)mkl_malloc(n * lda * sizeof(float), 64);
  std::generate(a, a + n * lda, std::bind(dist, gen));

  x0 = (float *)malloc((1 + (n - 1) * incx) * sizeof(float));
  x1 = (float *)mkl_malloc((1 + (n - 1) * incx) * sizeof(float), 64);
  std::generate(x0, x0 + (1 + (n - 1) * incx), std::bind(dist, gen));
  memcpy(x1, x0, (1 + (n - 1) * incx) * sizeof(float));

  const double start0 = dsecnd();
  naive_stbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, 0, a,
              lda, x0, incx);
  const double end0 = dsecnd();

  const double start1 = dsecnd();
  cblas_stbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, 0, a,
              lda, x1, incx);
  const double end1 = dsecnd();

  float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
  float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
  for (int i = 0; i < n; ++i) {
    const float err_abs = std::abs(x0[incx * i] - x1[incx * i]);
    const float err_rel = std::abs(err_abs / x0[incx * i]);
    err_abs_min = std::min(err_abs_min, err_abs);
    err_abs_max = std::max(err_abs_max, err_abs);
    err_rel_min = std::min(err_rel_min, err_rel);
    err_rel_max = std::max(err_rel_max, err_rel);
  }

  printf(
      "err_abs = [%f, %f], "
      "err_rel = [%f, %f], "
      "%f sec -> %f sec, "
      "%f Mflop/s -> %f Mflop/s\n",
      err_abs_min, err_abs_max, err_rel_min, err_rel_max, end0 - start0,
      end1 - start1, n / (end0 - start0) * 1e-6, n / (end1 - start1) * 1e-6);

  if (err_rel_max > 1e-3f) {
    std::cerr << "error: Absolute error is too large" << std::endl;
    return 1;
  }

  mkl_free(a);
  free(x0);
  mkl_free(x1);
  return 0;
}

template <class Generator>
static int test_stbmv_diag_random(Generator &gen) {
  std::uniform_int_distribution<int> dist_int;

  dist_int.param(decltype(dist_int)::param_type(1, 1 << 16));
  const int n = dist_int(gen);

  dist_int.param(decltype(dist_int)::param_type(1, 8));
  const int incx = dist_int(gen), lda = dist_int(gen);

  return test_stbmv_diag_single(n, lda, incx, gen);
}

int main(void) {
  setbuf(stdout, NULL);

  std::default_random_engine gen;
  int ret;

  for (int i = 0; i < 10; ++i) {
    ret = test_stbmv_diag_random(gen);
    if (ret) return ret;
  }

  return 0;
}
