#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>

#include "cblasdefs.h"
#include "cxxutils.hpp"

template <typename T>
static void naive_tbmv([[maybe_unused]] const CBLAS_LAYOUT layout,
                       [[maybe_unused]] const CBLAS_UPLO uplo,
                       [[maybe_unused]] const CBLAS_TRANSPOSE trans,
                       [[maybe_unused]] const CBLAS_DIAG diag, const int n,
                       const int k, const T *const a, const int lda, T *const x,
                       const int incx) {
  if (k != 0) {
    std::cerr << "error: We only support diagonal matrices for now"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

#pragma omp parallel for default(none) firstprivate(n, a, lda, x, incx) \
    schedule(guided)
  for (int i = 0; i < n; ++i) x[incx * i] *= a[lda * i];
}

template <typename T,
          typename std::enable_if_t<std::is_same<T, float>{}, bool> = true>
static void cblas_tbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                       const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                       const int n, const int k, const T *const a,
                       const int lda, T *const x, const int incx) {
  cblas_stbmv(layout, uplo, trans, diag, n, k, a, lda, x, incx);
}

template <typename T, typename std::enable_if_t<
                          std::is_same<T, std::complex<float>>{}, bool> = true>
static void cblas_tbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                       const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                       const int n, const int k, const T *const a,
                       const int lda, T *const x, const int incx) {
  cblas_ctbmv(layout, uplo, trans, diag, n, k, a, lda, x, incx);
}

template <typename T, class Generator>
static int test_tbmv_diag_single(const int n, const int lda, const int incx,
                                 Generator &gen) {
  T *a, *x0, *x1;
  uniform_distribution<T> dist;

  printf("%stbmv: n = %5d, lda = %5d, incx = %d: ", blas_prefix<T>(), n, lda,
         incx);

  a = (T *)mkl_malloc(n * lda * sizeof(T), 64);
  std::generate(a, a + n * lda, std::bind(dist, gen));

  x0 = (T *)malloc((1 + (n - 1) * incx) * sizeof(T));
  x1 = (T *)mkl_malloc((1 + (n - 1) * incx) * sizeof(T), 64);
  std::generate(x0, x0 + (1 + (n - 1) * incx), std::bind(dist, gen));
  memcpy(x1, x0, (1 + (n - 1) * incx) * sizeof(T));

  const double start0 = dsecnd();
  naive_tbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, 0, a,
             lda, x0, incx);
  const double t0 = dsecnd() - start0;

  const double start1 = dsecnd();
  cblas_tbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, 0, a,
             lda, x1, incx);
  const double t1 = dsecnd() - start1;

  using error_type = decltype(std::abs(T()));

  error_type err_abs_min = std::numeric_limits<error_type>::infinity(),
             err_abs_max = -std::numeric_limits<error_type>::infinity(),
             err_rel_min = std::numeric_limits<error_type>::infinity(),
             err_rel_max = -std::numeric_limits<error_type>::infinity();
  for (int i = 0; i < n; ++i) {
    const T err = x0[incx * i] - x1[incx * i];
    const error_type err_abs = std::abs(err),
                     err_rel = std::abs(err / x0[incx * i]);
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
      err_abs_min, err_abs_max, err_rel_min, err_rel_max, t0, t1, n / t0 * 1e-6,
      n / t1 * 1e-6);

  if (err_rel_max > 1e-3) {
    std::cerr << "error: Absolute error is too large" << std::endl;
    return 1;
  }

  mkl_free(a);
  free(x0);
  mkl_free(x1);
  return 0;
}

template <typename T, class Generator>
static int test_tbmv_diag_random(Generator &gen) {
  std::uniform_int_distribution<int> dist_int;

  dist_int.param(decltype(dist_int)::param_type(1, 1 << 16));
  const int n = dist_int(gen);

  dist_int.param(decltype(dist_int)::param_type(1, 8));
  const int incx = dist_int(gen), lda = dist_int(gen);

  return test_tbmv_diag_single<T>(n, lda, incx, gen);
}

int main(void) {
  setbuf(stdout, NULL);

  std::default_random_engine gen;
  int ret;

  for (int i = 0; i < 10; ++i) {
    ret = test_tbmv_diag_random<float>(gen);
    if (ret) return ret;
  }

  for (int i = 0; i < 10; ++i) {
    ret = test_tbmv_diag_random<std::complex<float>>(gen);
    if (ret) return ret;
  }

  return 0;
}
