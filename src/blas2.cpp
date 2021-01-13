#include <cstdint>
#include <cstdio>

#include "cblas-qmkl6.h"
#include "qmkl6-internal.hpp"

static const uint64_t qpu_sgemv_n_orig[] = {
#include "sgemv_n.qhex6"
};

static const uint64_t qpu_sgemv_t_orig[] = {
#include "sgemv_t.qhex6"
};

static const uint64_t qpu_stbmv_orig[] = {
#include "stbmv.qhex6"
};

void cblas_sgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans,
                 const int m, const int n, const float alpha, const float *a,
                 const int lda, const float *x, const int incx,
                 const float beta, float *y, const int incy) {
  if (m <= 0) {
    fprintf(stderr, "error: m (%d) must be greater than zero\n", m);
    XERBLA(1);
  }
  if (n <= 0) {
    fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
    XERBLA(1);
  }
  if (lda <= 0) {
    fprintf(stderr, "error: lda (%d) must be greater than zero\n", lda);
    XERBLA(1);
  }
  if (incx <= 0) {
    fprintf(stderr, "error: incx (%d) must be greater than zero\n", incx);
    XERBLA(1);
  }
  if (incy <= 0) {
    fprintf(stderr, "error: incy (%d) must be greater than zero\n", incy);
    XERBLA(1);
  }
  if (layout == CblasRowMajor) {
    if (lda < n) {
      fprintf(stderr,
              "error: lda (%d) must not be smaller than n (%d)"
              " for row major\n",
              lda, n);
      XERBLA(1);
    }
  } else {
    if (lda < m) {
      fprintf(stderr,
              "error: lda (%d) must not be smaller than m (%d)"
              " for column major\n",
              lda, m);
      XERBLA(1);
    }
  }

  if (trans != CblasNoTrans && trans != CblasTrans) {
    fprintf(stderr, "error: Only NoTrans and Trans are supported\n");
    XERBLA(1);
  }

  constexpr int num_qpus = 8, split_x = 64 * 16, split_y = num_qpus;
  const int xlen = (trans == CblasNoTrans) ? n : m;
  const int ylen = (trans == CblasNoTrans) ? m : n;
  const bool use_qpu = (xlen >= split_x) && (ylen >= split_y);
  const int xlen_qpu = use_qpu ? xlen / split_x * split_x : 0;
  const int ylen_qpu = use_qpu ? ylen / split_y * split_y : 0;

  uint32_t a_handle, x_handle, y_handle;
  uint32_t a_bus, x_bus, y_bus;

  if (use_qpu) {
    qmkl6.locate_virt((void *)a, a_handle, a_bus);
    qmkl6.locate_virt((void *)x, x_handle, x_bus);
    qmkl6.locate_virt((void *)y, y_handle, y_bus);

    qmkl6.unif[0] = incx;
    qmkl6.unif[1] = x_bus;
    qmkl6.unif[2] = incy;
    qmkl6.unif[3] = y_bus;
    qmkl6.unif[4] = lda;
    qmkl6.unif[5] = a_bus;
    if (layout == CblasRowMajor) {
      qmkl6.unif[6] = m;
      qmkl6.unif[7] = n;
    } else {
      qmkl6.unif[6] = n;
      qmkl6.unif[7] = m;
    }
    qmkl6.unif[8] = qmkl6.bit_cast<uint32_t>(alpha);
    qmkl6.unif[9] = qmkl6.bit_cast<uint32_t>(beta);

    qmkl6.execute_qpu_code((layout == CblasRowMajor) == (trans == CblasNoTrans)
                               ? qmkl6.qpu_sgemv_n_bus
                               : qmkl6.qpu_sgemv_t_bus,
                           qmkl6.unif_bus, num_qpus, 1, y_handle);
  }

  float x_host[xlen], y_host[ylen_qpu];

  for (int i = 0; i < xlen; ++i) x_host[i] = x[incx * i];

  if ((layout == CblasRowMajor) == (trans == CblasNoTrans)) {
    for (int i = 0; i < ylen_qpu; ++i, a += lda) {
      float s = .0f;
      for (int j = xlen_qpu; j < xlen; ++j) s += a[j] * x_host[j];
      y_host[i] = alpha * s;
    }
    for (int i = ylen_qpu; i < ylen; ++i, a += lda) {
      float s = .0f;
      for (int j = 0; j < xlen; ++j) s += a[j] * x_host[j];
      y[incy * i] = alpha * s + beta * y[incy * i];
    }
  } else {
    for (int i = 0; i < ylen_qpu; ++i, ++a) {
      float s = .0f;
      for (int j = xlen_qpu; j < xlen; ++j) s += a[lda * j] * x_host[j];
      y_host[i] = alpha * s;
    }
    for (int i = ylen_qpu; i < ylen; ++i, ++a) {
      float s = .0f;
      for (int j = 0; j < xlen; ++j) s += a[lda * j] * x_host[j];
      y[incy * i] = alpha * s + beta * y[incy * i];
    }
  }

  if (use_qpu) qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, y_handle);

  for (int i = 0; i < ylen_qpu; ++i) y[incy * i] += y_host[i];
}

void cblas_stbmv([[maybe_unused]] const CBLAS_LAYOUT layout,
                 [[maybe_unused]] const CBLAS_UPLO uplo,
                 [[maybe_unused]] const CBLAS_TRANSPOSE trans,
                 [[maybe_unused]] const CBLAS_DIAG diag, int n, const int k,
                 const float *const a, const int lda, float *const x,
                 const int incx) {
  if (n <= 0) {
    fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
    XERBLA(1);
  }
  if (k != 0) {
    fprintf(stderr, "error: Diagonal matrices are only supported for now\n");
    XERBLA(1);
  }
  if (lda <= 0) {
    fprintf(stderr, "error: lda (%d) must be greater than zero\n", lda);
    XERBLA(1);
  }
  if (incx <= 0) {
    fprintf(stderr, "error: incx (%d) must be greater than zero\n", incx);
    XERBLA(1);
  }

  const unsigned num_queues = 4, num_threads = 16, num_qpus = 8,
                 unroll = 1 << 1,
                 align = num_queues * num_threads * num_qpus * unroll;
  const int n_rem = n % align;
  n -= n_rem;

  uint32_t a_handle, x_handle, a_bus, x_bus;

  if (n > 0) {
    qmkl6.locate_virt((void *)a, a_handle, a_bus);
    qmkl6.locate_virt((void *)x, x_handle, x_bus);

    qmkl6.unif[0] = n;
    qmkl6.unif[1] = a_bus;
    qmkl6.unif[2] = lda;
    qmkl6.unif[3] = x_bus;
    qmkl6.unif[4] = incx;

    qmkl6.execute_qpu_code(qmkl6.qpu_stbmv_bus, qmkl6.unif_bus, num_qpus, 1,
                           a_handle);
  }

  for (int i = 0, j = lda * n, k = incx * n; i < n_rem;
       ++i, j += lda, k += incx)
    x[k] *= a[j];

  if (n > 0) qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, a_handle);
}

void qmkl6_context::init_blas2(void) {
  qpu_sgemv_n = (uint64_t *)alloc_memory(sizeof(qpu_sgemv_n_orig),
                                         qpu_sgemv_n_handle, qpu_sgemv_n_bus);
  memcpy(qpu_sgemv_n, qpu_sgemv_n_orig, sizeof(qpu_sgemv_n_orig));

  qpu_sgemv_t = (uint64_t *)alloc_memory(sizeof(qpu_sgemv_t_orig),
                                         qpu_sgemv_t_handle, qpu_sgemv_t_bus);
  memcpy(qpu_sgemv_t, qpu_sgemv_t_orig, sizeof(qpu_sgemv_t_orig));

  qpu_stbmv = (uint64_t *)alloc_memory(sizeof(qpu_stbmv_orig), qpu_stbmv_handle,
                                       qpu_stbmv_bus);
  memcpy(qpu_stbmv, qpu_stbmv_orig, sizeof(qpu_stbmv_orig));
}

void qmkl6_context::finalize_blas2(void) {
  free_memory(sizeof(qpu_stbmv_orig), qpu_stbmv_handle, qpu_stbmv);
  free_memory(sizeof(qpu_sgemv_t_orig), qpu_sgemv_t_handle, qpu_sgemv_t);
  free_memory(sizeof(qpu_sgemv_n_orig), qpu_sgemv_n_handle, qpu_sgemv_n);
}
