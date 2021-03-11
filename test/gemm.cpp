#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>

#include "cblasdefs.h"

static void naive_sgemm_rnn(const int M, const int N, const int K,
                            const float alpha, const float *A, const int lda,
                            const float *B, const int ldb, const float beta,
                            float *C, const int ldc) {
#if 0

#pragma omp parallel for schedule(static) default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = .0f;
            for (int k = 0; k < K; ++k)
                s += A[lda * m + k] * B[ldb * k + n];
            C[ldc * m + n] = alpha * s + beta * C[ldc * m + n];
        }
    }

#else

#pragma omp parallel default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
  {
    constexpr int split_m = 32, split_n = 32, split_k = 32;
    float aa[split_m][split_k], bb[split_n][split_k];
#pragma omp for collapse(2) schedule(guided)
    for (int m = 0; m < M; m += split_m) {
      for (int n = 0; n < N; n += split_n) {
        const int bound_mm = (M - m < split_m) ? M - m : split_m;
        const int bound_nn = (N - n < split_n) ? N - n : split_n;
        for (int mm = 0; mm < bound_mm; ++mm)
          for (int nn = 0; nn < bound_nn; ++nn)
            C[ldc * m + n + ldc * mm + nn] *= beta;
        for (int k = 0; k < K; k += split_k) {
          const int bound_kk = (K - k < split_k) ? K - k : split_k;
          for (int mm = 0; mm < bound_mm; ++mm)
            for (int kk = 0; kk < bound_kk; ++kk)
              aa[mm][kk] = A[lda * m + k + lda * mm + kk];
          for (int kk = 0; kk < bound_kk; ++kk)
            for (int nn = 0; nn < bound_nn; ++nn)
              bb[nn][kk] = B[ldb * k + n + ldb * kk + nn];
          for (int mm = 0; mm < bound_mm; ++mm) {
            for (int nn = 0; nn < bound_nn; ++nn) {
              float s = 0.f;
              for (int kk = 0; kk < bound_kk; ++kk)
                s += aa[mm][kk] * bb[nn][kk];
              C[ldc * m + n + ldc * mm + nn] += alpha * s;
            }
          }
        }
      }
    }
  }

#endif
}

static void naive_sgemm_rnt(const int M, const int N, const int K,
                            const float alpha, const float *A, const int lda,
                            const float *B, const int ldb, const float beta,
                            float *C, const int ldc) {
#if 0

#pragma omp parallel for schedule(static) default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = .0f;
            for (int k = 0; k < K; ++k)
                s += A[lda * m + k] * B[ldb * n + k];
            C[ldc * m + n] = alpha * s + beta * C[ldc * m + n];
        }
    }

#else

#pragma omp parallel default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
  {
    constexpr int split_m = 16, split_n = 16, split_k = 64;
    float aa[split_m][split_k], bb[split_n][split_k];
#pragma omp for collapse(2) schedule(guided)
    for (int m = 0; m < M; m += split_m) {
      for (int n = 0; n < N; n += split_n) {
        const int bound_mm = (M - m < split_m) ? M - m : split_m;
        const int bound_nn = (N - n < split_n) ? N - n : split_n;
        for (int mm = 0; mm < bound_mm; ++mm)
          for (int nn = 0; nn < bound_nn; ++nn)
            C[ldc * m + n + ldc * mm + nn] *= beta;
        for (int k = 0; k < K; k += split_k) {
          const int bound_kk = (K - k < split_k) ? K - k : split_k;
          for (int mm = 0; mm < bound_mm; ++mm)
            for (int kk = 0; kk < bound_kk; ++kk)
              aa[mm][kk] = A[lda * m + k + lda * mm + kk];
          for (int nn = 0; nn < bound_nn; ++nn)
            for (int kk = 0; kk < bound_kk; ++kk)
              bb[nn][kk] = B[ldb * n + k + ldb * nn + kk];
          for (int mm = 0; mm < bound_mm; ++mm) {
            for (int nn = 0; nn < bound_nn; ++nn) {
              float s = 0.f;
              for (int kk = 0; kk < bound_kk; ++kk)
                s += aa[mm][kk] * bb[nn][kk];
              C[ldc * m + n + ldc * mm + nn] += alpha * s;
            }
          }
        }
      }
    }
  }

#endif
}

static void naive_sgemm_rtn(const int M, const int N, const int K,
                            const float alpha, const float *A, const int lda,
                            const float *B, const int ldb, const float beta,
                            float *C, const int ldc) {
#if 0

#pragma omp parallel for schedule(static) default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = .0f;
            for (int k = 0; k < K; ++k)
                s += A[lda * k + m] * B[ldb * k + n];
            C[ldc * m + n] = alpha * s + beta * C[ldc * m + n];
        }
    }

#else

#pragma omp parallel default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
  {
    constexpr int split_m = 32, split_n = 32, split_k = 32;
    float aa[split_m][split_k], bb[split_n][split_k];
#pragma omp for collapse(2) schedule(guided)
    for (int m = 0; m < M; m += split_m) {
      for (int n = 0; n < N; n += split_n) {
        const int bound_mm = (M - m < split_m) ? M - m : split_m;
        const int bound_nn = (N - n < split_n) ? N - n : split_n;
        for (int mm = 0; mm < bound_mm; ++mm)
          for (int nn = 0; nn < bound_nn; ++nn)
            C[ldc * m + n + ldc * mm + nn] *= beta;
        for (int k = 0; k < K; k += split_k) {
          const int bound_kk = (K - k < split_k) ? K - k : split_k;
          for (int kk = 0; kk < bound_kk; ++kk)
            for (int mm = 0; mm < bound_mm; ++mm)
              aa[mm][kk] = A[lda * k + m + lda * kk + mm];
          for (int kk = 0; kk < bound_kk; ++kk)
            for (int nn = 0; nn < bound_nn; ++nn)
              bb[nn][kk] = B[ldb * k + n + ldb * kk + nn];
          for (int mm = 0; mm < bound_mm; ++mm) {
            for (int nn = 0; nn < bound_nn; ++nn) {
              float s = 0.f;
              for (int kk = 0; kk < bound_kk; ++kk)
                s += aa[mm][kk] * bb[nn][kk];
              C[ldc * m + n + ldc * mm + nn] += alpha * s;
            }
          }
        }
      }
    }
  }

#endif
}

static void naive_sgemm_rtt(const int M, const int N, const int K,
                            const float alpha, const float *A, const int lda,
                            const float *B, const int ldb, const float beta,
                            float *C, const int ldc) {
#if 0

#pragma omp parallel for schedule(static) default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = .0f;
            for (int k = 0; k < K; ++k)
                s += A[lda * k + m] * B[ldb * n + k];
            C[ldc * m + n] = alpha * s + beta * C[ldc * m + n];
        }
    }

#else

#pragma omp parallel default(none) \
    firstprivate(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
  {
    constexpr int split_m = 32, split_n = 32, split_k = 32;
    float aa[split_m][split_k], bb[split_n][split_k];
#pragma omp for collapse(2) schedule(guided)
    for (int m = 0; m < M; m += split_m) {
      for (int n = 0; n < N; n += split_n) {
        const int bound_mm = (M - m < split_m) ? M - m : split_m;
        const int bound_nn = (N - n < split_n) ? N - n : split_n;
        for (int mm = 0; mm < bound_mm; ++mm)
          for (int nn = 0; nn < bound_nn; ++nn)
            C[ldc * m + n + ldc * mm + nn] *= beta;
        for (int k = 0; k < K; k += split_k) {
          const int bound_kk = (K - k < split_k) ? K - k : split_k;
          for (int kk = 0; kk < bound_kk; ++kk)
            for (int mm = 0; mm < bound_mm; ++mm)
              aa[mm][kk] = A[lda * k + m + lda * kk + mm];
          for (int nn = 0; nn < bound_nn; ++nn)
            for (int kk = 0; kk < bound_kk; ++kk)
              bb[nn][kk] = B[ldb * n + k + ldb * nn + kk];
          for (int mm = 0; mm < bound_mm; ++mm) {
            for (int nn = 0; nn < bound_nn; ++nn) {
              float s = 0.f;
              for (int kk = 0; kk < bound_kk; ++kk)
                s += aa[mm][kk] * bb[nn][kk];
              C[ldc * m + n + ldc * mm + nn] += alpha * s;
            }
          }
        }
      }
    }
  }

#endif
}

static void naive_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
                        const CBLAS_TRANSPOSE transb, const int m, const int n,
                        const int k, const float alpha, const float *a,
                        const int lda, const float *b, const int ldb,
                        const float beta, float *c, const int ldc) {
  if (layout == CblasRowMajor && transa == CblasNoTrans &&
      transb == CblasNoTrans)
    naive_sgemm_rnn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  else if (layout == CblasRowMajor && transa == CblasNoTrans &&
           transb == CblasTrans)
    naive_sgemm_rnt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  else if (layout == CblasRowMajor && transa == CblasTrans &&
           transb == CblasNoTrans)
    naive_sgemm_rtn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  else if (layout == CblasRowMajor && transa == CblasTrans &&
           transb == CblasTrans)
    naive_sgemm_rtt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  else if (layout == CblasColMajor && transa == CblasNoTrans &&
           transb == CblasNoTrans)
    naive_sgemm_rnn(n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
  else if (layout == CblasColMajor && transa == CblasNoTrans &&
           transb == CblasTrans)
    naive_sgemm_rtn(n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
  else if (layout == CblasColMajor && transa == CblasTrans &&
           transb == CblasNoTrans)
    naive_sgemm_rnt(n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
  else if (layout == CblasColMajor && transa == CblasTrans &&
           transb == CblasTrans)
    naive_sgemm_rtt(n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
  else {
    std::cerr << "error: Unsupported layout and trans" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <class Generator>
static int test_sgemm_single(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const int m,
                             const int n, const int k, const float alpha,
                             const int lda, const int ldb, const float beta,
                             const int ldc, Generator &gen) {
  float *A, *B, *C0, *C1;
  std::uniform_real_distribution<float> dist;

  printf(
      "%c%c%c: m = %5d, n = %5d, k = %5d, alpha = %f, lda = %5d, "
      "ldb = %5d, beta = %f, ldc = %5d: ",
      (layout == CblasRowMajor) ? 'R' : 'C',
      (transa == CblasNoTrans) ? 'N' : 'T',
      (transb == CblasNoTrans) ? 'N' : 'T', m, n, k, alpha, lda, ldb, beta,
      ldc);

  if ((layout == CblasRowMajor) == (transa == CblasNoTrans)) {
    if (lda < k) {
      std::cerr << "error: lda must be >= k for row major, no trans type"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    A = (float *)mkl_malloc(m * lda * sizeof(float), 64);
    std::generate(A, A + m * lda, std::bind(dist, gen));
  } else {
    if (lda < m) {
      std::cerr << "error: lda must be >= m for row major, trans type"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    A = (float *)mkl_malloc(k * lda * sizeof(float), 64);
    std::generate(A, A + k * lda, std::bind(dist, gen));
  }

  if ((layout == CblasRowMajor) == (transb == CblasNoTrans)) {
    if (ldb < n) {
      std::cerr << "error: ldb must be >= n for row major, no trans type"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    B = (float *)mkl_malloc(k * ldb * sizeof(float), 64);
    std::generate(B, B + k * ldb, std::bind(dist, gen));
  } else {
    if (ldb < k) {
      std::cerr << "error: ldb must be >= k for row major, trans type"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    B = (float *)mkl_malloc(n * ldb * sizeof(float), 64);
    std::generate(B, B + n * ldb, std::bind(dist, gen));
  }

  if (layout == CblasRowMajor) {
    if (ldc < n) {
      std::cerr << "error: ldc must be >= n for row major" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    C0 = (float *)malloc(m * ldc * sizeof(float));
    C1 = (float *)mkl_malloc(m * ldc * sizeof(float), 64);
    std::generate(C0, C0 + m * ldc, std::bind(dist, gen));
    memcpy(C1, C0, m * ldc * sizeof(float));
  } else {
    if (ldc < m) {
      std::cerr << "error: ldc must be >= m for col major" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    C0 = (float *)malloc(n * ldc * sizeof(float));
    C1 = (float *)mkl_malloc(n * ldc * sizeof(float), 64);
    std::generate(C0, C0 + n * ldc, std::bind(dist, gen));
    memcpy(C1, C0, n * ldc * sizeof(float));
  }

  naive_sgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C0,
              ldc);

  const double start = dsecnd();
  cblas_sgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C1,
              ldc);
  const double end = dsecnd();

  float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
  float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
  if (layout == CblasRowMajor) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        const float err_abs = std::abs(C0[ldc * i + j] - C1[ldc * i + j]);
        const float err_rel = std::abs(err_abs / C0[ldc * i + j]);
        err_abs_min = std::min(err_abs_min, err_abs);
        err_abs_max = std::max(err_abs_max, err_abs);
        err_rel_min = std::min(err_rel_min, err_rel);
        err_rel_max = std::max(err_rel_max, err_rel);
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        const float err_abs = std::abs(C0[ldc * i + j] - C1[ldc * i + j]);
        const float err_rel = std::abs(err_abs / C0[ldc * i + j]);
        err_abs_min = std::min(err_abs_min, err_abs);
        err_abs_max = std::max(err_abs_max, err_abs);
        err_rel_min = std::min(err_rel_min, err_rel);
        err_rel_max = std::max(err_rel_max, err_rel);
      }
    }
  }

  printf("err_abs = [%f, %f], err_rel = [%f, %f], %f sec, %f Gflop/s\n",
         err_abs_min, err_abs_max, err_rel_min, err_rel_max, end - start,
         (2. * k * n * m + 3. * n * m) / (end - start) * 1e-9);

  if (err_rel_max > 1e-3f) {
    std::cerr << "error: Absolute error is too large\n" << std::endl;
    return 1;
  }

  mkl_free(A);
  mkl_free(B);
  free(C0);
  mkl_free(C1);
  return 0;
}

template <class Generator>
static int test_sgemm_random(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, Generator &gen) {
  std::uniform_int_distribution<int> dist_int;
  std::uniform_real_distribution<float> dist_value;

  dist_int.param(decltype(dist_int)::param_type(1, 1 << 11));
  const int m = dist_int(gen), n = dist_int(gen), k = dist_int(gen);

  if ((layout == CblasRowMajor) == (transa == CblasNoTrans))
    dist_int.param(decltype(dist_int)::param_type(k, k * 4));
  else
    dist_int.param(decltype(dist_int)::param_type(m, m * 4));
  const int lda = dist_int(gen);

  if ((layout == CblasRowMajor) == (transb == CblasNoTrans))
    dist_int.param(decltype(dist_int)::param_type(n, n * 4));
  else
    dist_int.param(decltype(dist_int)::param_type(k, k * 4));
  const int ldb = dist_int(gen);

  if (layout == CblasRowMajor)
    dist_int.param(decltype(dist_int)::param_type(n, n * 4));
  else
    dist_int.param(decltype(dist_int)::param_type(m, m * 4));
  const int ldc = dist_int(gen);

  const float alpha = dist_value(gen), beta = dist_value(gen);

  return test_sgemm_single(layout, transa, transb, m, n, k, alpha, lda, ldb,
                           beta, ldc, gen);
}

int main(void) {
  setbuf(stdout, NULL);

  std::default_random_engine gen;
  int ret;

  for (CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    const int m = 512, n = 512, k = 1024;
    const int ldc = (layout == CblasRowMajor) ? n : m;
    const float alpha = 2.f, beta = 3.f;
    for (CBLAS_TRANSPOSE transa : {CblasNoTrans, CblasTrans}) {
      const int lda =
          ((layout == CblasRowMajor) == (transa == CblasNoTrans)) ? k : m;
      for (CBLAS_TRANSPOSE transb : {CblasNoTrans, CblasTrans}) {
        const int ldb =
            ((layout == CblasRowMajor) == (transb == CblasNoTrans)) ? n : k;
        ret = test_sgemm_single(layout, transa, transb, m, n, k, alpha, lda,
                                ldb, beta, ldc, gen);
        if (ret) return ret;
      }
    }
  }

  for (CBLAS_LAYOUT layout : {CblasRowMajor, CblasColMajor}) {
    for (CBLAS_TRANSPOSE transa : {CblasNoTrans, CblasTrans}) {
      for (CBLAS_TRANSPOSE transb : {CblasNoTrans, CblasTrans}) {
        for (int i = 0; i < 5; ++i) {
          ret = test_sgemm_random(layout, transa, transb, gen);
          if (ret) return ret;
        }
      }
    }
  }

  return 0;
}
