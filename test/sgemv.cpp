#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>

#include "cblasdefs.h"

static void naive_sgemv_n(const int m, const int n, const float alpha,
                          const float *a, const int lda, const float *const x,
                          const int incx, const float beta, float *const y,
                          const int incy) {
  for (int i = 0; i < m; ++i, a += lda) {
    float s = .0f;
    for (int j = 0; j < n; ++j) s += a[j] * x[incx * j];
    y[incy * i] = alpha * s + beta * y[incy * i];
  }
}

static void naive_sgemv_t(const int m, const int n, const float alpha,
                          const float *a, const int lda, const float *const x,
                          const int incx, const float beta, float *const y,
                          const int incy) {
  for (int i = 0; i < n; ++i, ++a) {
    float s = .0f;
    for (int j = 0; j < m; ++j) s += a[lda * j] * x[incx * j];
    y[incy * i] = alpha * s + beta * y[incy * i];
  }
}

static void naive_sgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans,
                        const int m, const int n, const float alpha,
                        const float *a, const int lda, const float *const x,
                        const int incx, const float beta, float *const y,
                        const int incy) {
  if (layout == CblasRowMajor && trans == CblasNoTrans)
    naive_sgemv_n(m, n, alpha, a, lda, x, incx, beta, y, incy);
  else if (layout == CblasRowMajor && trans == CblasTrans)
    naive_sgemv_t(m, n, alpha, a, lda, x, incx, beta, y, incy);
  else if (layout == CblasColMajor && trans == CblasNoTrans)
    naive_sgemv_t(n, m, alpha, a, lda, x, incx, beta, y, incy);
  else if (layout == CblasColMajor && trans == CblasTrans)
    naive_sgemv_n(n, m, alpha, a, lda, x, incx, beta, y, incy);
  else {
    std::cerr << "error: Unsupported layout and trans" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <class Generator>
static int test_sgemv_single(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const int m,
                             const int n, const float alpha, const int lda,
                             const int incx, const float beta, const int incy,
                             Generator &gen) {
  float *a, *x, *y0, *y1;
  std::uniform_real_distribution<float> dist;

  printf(
      "layout = %s, trans = %7s, m = %5d, n = %5d, alpha = %f, lda = %5d, "
      "incx = %d, beta = %f, incy = %d: ",
      (layout == CblasRowMajor) ? "RowMajor" : "ColMajor",
      (trans == CblasNoTrans) ? "NoTrans" : "Trans", m, n, alpha, lda, incx,
      beta, incy);

  if (layout == CblasRowMajor) {
    if (lda < n) {
      std::cerr << "error: lda must be >= n for row major" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    a = (float *)mkl_malloc(m * lda * sizeof(float), 64);
    std::generate(a, a + m * lda, std::bind(dist, gen));
  } else {
    if (lda < m) {
      std::cerr << "error: lda must be >= m for column major" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    a = (float *)mkl_malloc(n * lda * sizeof(float), 64);
    std::generate(a, a + n * lda, std::bind(dist, gen));
  }

  const int xlen = (trans == CblasNoTrans) ? n : m;
  const int ylen = (trans == CblasNoTrans) ? m : n;

  x = (float *)mkl_malloc((1 + (xlen - 1) * incx) * sizeof(float), 64);
  y0 = (float *)malloc((1 + (ylen - 1) * incy) * sizeof(float));
  y1 = (float *)mkl_malloc((1 + (ylen - 1) * incy) * sizeof(float), 64);
  std::generate(x, x + (1 + (xlen - 1) * incx), std::bind(dist, gen));
  std::generate(y0, y0 + (1 + (ylen - 1) * incy), std::bind(dist, gen));
  memcpy(y1, y0, (1 + (ylen - 1) * incy) * sizeof(float));

  naive_sgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y0, incy);

  const double start = dsecond();
  cblas_sgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y1, incy);
  const double end = dsecond();

  float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
  float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
  for (int i = 0; i < ylen; ++i) {
    const float err_abs = std::abs(y0[incy * i] - y1[incy * i]);
    const float err_rel = std::abs(err_abs / y0[incy * i]);
    err_abs_min = std::min(err_abs_min, err_abs);
    err_abs_max = std::max(err_abs_max, err_abs);
    err_rel_min = std::min(err_rel_min, err_rel);
    err_rel_max = std::max(err_rel_max, err_rel);
  }

  printf("err_abs = [%f, %f], err_rel = [%f, %f], %f sec, %f Mflop/s\n",
         err_abs_min, err_abs_max, err_rel_min, err_rel_max, end - start,
         (2. * xlen * ylen + ylen) / (end - start) * 1e-6);

  if (err_rel_max > 1e-3f) {
    std::cerr << "error: Absolute error is too large" << std::endl;
    return 1;
  }

  mkl_free(a);
  mkl_free(x);
  free(y0);
  mkl_free(y1);
  return 0;
}

template <class Generator>
static int test_sgemv_random(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, Generator &gen) {
  std::uniform_int_distribution<int> dist_int;
  std::uniform_real_distribution<float> dist_value;

  dist_int.param(decltype(dist_int)::param_type(1, 1 << 11));
  const int m = dist_int(gen), n = dist_int(gen);

  dist_int.param(decltype(dist_int)::param_type(1, 8));
  const int incx = dist_int(gen), incy = dist_int(gen);

  if (layout == CblasRowMajor)
    dist_int.param(decltype(dist_int)::param_type(n, n * 4));
  else
    dist_int.param(decltype(dist_int)::param_type(m, m * 4));
  const int lda = dist_int(gen);

  const float alpha = dist_value(gen), beta = dist_value(gen);

  return test_sgemv_single(layout, trans, m, n, alpha, lda, incx, beta, incy,
                           gen);
}

int main(void) {
  setbuf(stdout, NULL);

  std::default_random_engine gen;
  int ret;

  for (CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    for (CBLAS_TRANSPOSE trans : {CblasNoTrans, CblasTrans}) {
      const int m = 1024, n = 1024;
      /* An extra 8-byte space to avoid collision misses. */
      const int lda = (layout == CblasRowMajor) ? n + 8 : m + 8;
      ret = test_sgemv_single(layout, trans, m, n, 1.f, lda, 1, 1.f, 1, gen);
      if (ret) return ret;
    }
  }

  for (CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    for (CBLAS_TRANSPOSE trans : {CblasNoTrans, CblasTrans}) {
      for (int i = 0; i < 10; ++i) {
        ret = test_sgemv_random(layout, trans, gen);
        if (ret) return ret;
      }
    }
  }

  return 0;
}
