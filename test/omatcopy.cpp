#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "cblasdefs.h"
#include "cxxutils.hpp"

template <typename T>
static void naive_omatcopy_n(const int rows, const int cols,
                             const T* const alpha, const T* const a,
                             const int lda, T* const b, const int ldb) {
#pragma omp parallel default(none) \
    firstprivate(rows, cols, alpha, a, lda, b, ldb) num_threads(1)
  {
    constexpr int block_rows = 8, block_cols = std::numeric_limits<int>::max();

#pragma omp for collapse(1) schedule(guided)
    for (int i = 0; i < rows; i += block_rows) {
      for (int j = 0; j < cols; j += block_cols) {
        const int bound_ii = (rows - i < block_rows) ? (rows - i) : block_rows,
                  bound_jj = (cols - j < block_cols) ? (cols - j) : block_cols;
        for (int ii = 0; ii < bound_ii; ++ii)
          for (int jj = 0; jj < bound_jj; ++jj)
            b[ldb * i + j + ldb * ii + jj] =
                *alpha * a[lda * i + j + lda * ii + jj];
      }
    }
  }
}

template <typename T>
static void naive_omatcopy_t(const int rows, const int cols,
                             const T* const alpha, const T* const a,
                             const int lda, T* const b, const int ldb) {
#pragma omp parallel default(none) \
    firstprivate(rows, cols, alpha, a, lda, b, ldb)
  {
    constexpr int block_rows = 32, block_cols = 32;
    T block[block_rows * block_cols];

#pragma omp for collapse(2) schedule(guided)
    for (int i = 0; i < rows; i += block_rows) {
      for (int j = 0; j < cols; j += block_cols) {
        const int bound_ii = (rows - i < block_rows) ? (rows - i) : block_rows,
                  bound_jj = (cols - j < block_cols) ? (cols - j) : block_cols,
                  ld = block_cols;
        for (int ii = 0; ii < bound_ii; ++ii)
          for (int jj = 0; jj < bound_jj; ++jj)
            block[ld * ii + jj] = *alpha * a[lda * i + j + lda * ii + jj];
        for (int jj = 0; jj < bound_jj; ++jj)
          for (int ii = 0; ii < bound_ii; ++ii)
            b[ldb * j + i + ldb * jj + ii] = block[ld * ii + jj];
      }
    }
  }
}

template <typename T>
static void naive_omatcopy(const CBLAS_LAYOUT layout,
                           const CBLAS_TRANSPOSE trans, const int rows,
                           const int cols, const T* const alpha,
                           const T* const a, const int lda, T* const b,
                           const int ldb) {
  if (layout == CblasRowMajor && trans == CblasNoTrans)
    naive_omatcopy_n(rows, cols, alpha, a, lda, b, ldb);
  else if (layout == CblasRowMajor && trans == CblasTrans)
    naive_omatcopy_t(rows, cols, alpha, a, lda, b, ldb);
  else if (layout == CblasColMajor && trans == CblasNoTrans)
    naive_omatcopy_n(cols, rows, alpha, a, lda, b, ldb);
  else if (layout == CblasColMajor && trans == CblasTrans)
    naive_omatcopy_t(cols, rows, alpha, a, lda, b, ldb);
  else {
    std::cerr << "error: Unsupported layout and trans" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <typename T,
          typename std::enable_if_t<std::is_same<T, float>{}, bool> = true>
static void cblas_omatcopy(const CBLAS_LAYOUT layout,
                           const CBLAS_TRANSPOSE trans, const int rows,
                           const int cols, const T* const alpha,
                           const T* const a, const int lda, T* const b,
                           const int ldb) {
  cblas_somatcopy(layout, trans, rows, cols, *alpha, a, lda, b, ldb);
}

template <typename T, typename std::enable_if_t<
                          std::is_same<T, std::complex<float>>{}, bool> = true>
static void cblas_omatcopy(const CBLAS_LAYOUT layout,
                           const CBLAS_TRANSPOSE trans, const int rows,
                           const int cols, const T* const alpha,
                           const T* const a, const int lda, T* const b,
                           const int ldb) {
  cblas_comatcopy(layout, trans, rows, cols, (const float*)alpha,
                  (const float*)a, lda, (float*)b, ldb);
}

template <typename T, class Generator>
static int test_omatcopy_single(const CBLAS_LAYOUT layout,
                                const CBLAS_TRANSPOSE trans, const int rows,
                                const int cols, const int lda, const int ldb,
                                Generator& gen) {
  uniform_distribution<T> dist;
  const T alpha = dist(gen);
  int a_len, b_len;
  T *a_host, *a_qpu, *b_host, *b_qpu;

  printf("%somatcopy %c%c: rows = %5d, cols = %5d, lda = %5d, ldb = %5d: ",
         blas_prefix<T>(), (layout == CblasRowMajor) ? 'R' : 'C',
         (trans == CblasNoTrans) ? 'N' : 'T', rows, cols, lda, ldb);

  if (layout == CblasRowMajor) {
    if (lda < cols) {
      std::cerr << "error: lda must be >= cols for row major" << std::endl;
      return 1;
    }
    a_len = rows * lda;
  } else {
    if (lda < rows) {
      std::cerr << "error: lda must be >= rows for col major" << std::endl;
      return 1;
    }
    a_len = cols * lda;
  }

  if ((layout == CblasRowMajor) == (trans == CblasNoTrans)) {
    if (ldb < cols) {
      std::cerr << "error: ldb must be >= cols for row major, no trans type"
                << std::endl;
      return 1;
    }
    b_len = rows * ldb;
  } else {
    if (ldb < rows) {
      std::cerr << "error: ldb must be >= rows for row major, trans type"
                << std::endl;
      return 1;
    }
    b_len = cols * ldb;
  }

  a_host = new T[a_len];
  a_qpu = (T*)mkl_malloc(a_len * sizeof(T), 64);
  std::generate(a_host, a_host + a_len, std::bind(dist, gen));
  std::copy(a_host, a_host + a_len, a_qpu);

  b_host = new T[b_len];
  b_qpu = (T*)mkl_malloc(b_len * sizeof(T), 64);
  std::generate(b_host, b_host + b_len, std::bind(dist, gen));
  std::copy(b_host, b_host + b_len, b_qpu);

  const double start_host = dsecnd();
  naive_omatcopy(layout, trans, rows, cols, &alpha, a_host, lda, b_host, ldb);
  const double time_host = dsecnd() - start_host;

  const double start_qpu = dsecnd();
  cblas_omatcopy(layout, trans, rows, cols, &alpha, a_qpu, lda, b_qpu, ldb);
  const double time_qpu = dsecnd() - start_qpu;

  using error_type = decltype(std::abs(T()));

  error_type err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
  error_type err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
  for (int i = 0; i < b_len; ++i) {
    const T err = b_host[i] - b_qpu[i];
    const error_type err_abs = std::abs(err),
                     err_rel = std::abs(err / b_host[i]);
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
      err_abs_min, err_abs_max, err_rel_min, err_rel_max, time_host, time_qpu,
      rows * cols / time_host * 1e-6, rows * cols / time_qpu * 1e-6);

  if (err_rel_max >= 1e-3f) {
    std::cerr << "error: Relative error is too large" << std::endl;
    return 1;
  }

  delete[] a_host;
  delete[] b_host;
  mkl_free(a_qpu);
  mkl_free(b_qpu);
  return 0;
}

template <typename T, class Generator>
static int test_omatcopy_random(const CBLAS_LAYOUT layout,
                                const CBLAS_TRANSPOSE trans, Generator& gen) {
  std::uniform_int_distribution<int> dist;

  const int dim_mask = (trans == CblasTrans) ? ~3 : ~0;

  dist.param(decltype(dist)::param_type(1, 1 << 11));
  const int rows = dist(gen) & dim_mask, cols = dist(gen) & dim_mask;

  if (layout == CblasRowMajor)
    dist.param(decltype(dist)::param_type(cols, cols * 4));
  else
    dist.param(decltype(dist)::param_type(rows, rows * 4));
  const int lda = dist(gen);

  if ((layout == CblasRowMajor) == (trans == CblasNoTrans))
    dist.param(decltype(dist)::param_type(cols, cols * 4));
  else
    dist.param(decltype(dist)::param_type(rows, rows * 4));
  const int ldb = dist(gen);

  return test_omatcopy_single<T>(layout, trans, rows, cols, lda, ldb, gen);
}

int main(void) {
  std::setbuf(stdout, NULL);

  std::default_random_engine gen;
  int ret;

  for (const CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    const int rows = 2048, cols = 4096;
    const int lda = (layout == CblasRowMajor) ? cols : rows;
    for (const CBLAS_TRANSPOSE trans : {CblasNoTrans, CblasTrans}) {
      const int ldb =
          ((layout == CblasRowMajor) == (trans == CblasNoTrans)) ? cols : rows;
      ret =
          test_omatcopy_single<float>(layout, trans, rows, cols, lda, ldb, gen);
      if (ret) return ret;
    }
  }

  for (const CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    const int rows = 2048, cols = 4096;
    const int lda = (layout == CblasRowMajor) ? cols : rows;
    for (const CBLAS_TRANSPOSE trans : {CblasNoTrans, CblasTrans}) {
      const int ldb =
          ((layout == CblasRowMajor) == (trans == CblasNoTrans)) ? cols : rows;
      ret = test_omatcopy_single<std::complex<float>>(layout, trans, rows, cols,
                                                      lda, ldb, gen);
      if (ret) return ret;
    }
  }

  for (const CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    for (const CBLAS_TRANSPOSE trans : {CblasNoTrans, CblasTrans}) {
      for (int i = 0; i < 5; ++i) {
        ret = test_omatcopy_random<float>(layout, trans, gen);
        if (ret) return ret;
      }
    }
  }

  for (const CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    for (const CBLAS_TRANSPOSE trans : {CblasNoTrans, CblasTrans}) {
      for (int i = 0; i < 5; ++i) {
        ret = test_omatcopy_random<std::complex<float>>(layout, trans, gen);
        if (ret) return ret;
      }
    }
  }

  return 0;
}
