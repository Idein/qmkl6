#include <cstdint>
#include <cstdio>
#include <cstring>

#include "cblas-qmkl6.h"
#include "qmkl6-internal.hpp"

/* somatcopy: B = a * op(A) */

static const uint64_t qpu_somatcopy_n_orig[] = {
#include "somatcopy_n.qhex6"
};

static const uint64_t qpu_somatcopy_t_4x4_orig[] = {
#include "somatcopy_t_4x4.qhex6"
};

static const uint64_t qpu_somatcopy_t_256x32_orig[] = {
#include "somatcopy_t_256x32.qhex6"
};

void cblas_somatcopy(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans,
                     const int rows, const int cols, const float alpha,
                     const float* const a, const int lda, float* const b,
                     const int ldb) {
  constexpr int num_qpus = 8;

  if (trans != CblasNoTrans && trans != CblasTrans) {
    fprintf(stderr,
            "error: trans must be NoTrans or Trans for single precision\n");
    XERBLA(1);
  }
  if (rows <= 0) {
    fprintf(stderr, "error: rows (%d) must be greater than zero\n", rows);
    XERBLA(1);
  }
  if (cols <= 0) {
    fprintf(stderr, "error: cols (%d) must be greater than zero\n", cols);
    XERBLA(1);
  }
  if (layout == CblasRowMajor) {
    if (lda < cols) {
      fprintf(stderr,
              "error: lda (%d) must not be smaller than cols (%d)"
              " for row major\n",
              lda, cols);
      XERBLA(1);
    }
  } else {
    if (lda < rows) {
      fprintf(stderr,
              "error: lda (%d) must not be smaller than rows (%d)"
              " for column major\n",
              lda, rows);
      XERBLA(1);
    }
  }
  if ((layout == CblasRowMajor) == (trans == CblasNoTrans)) {
    if (ldb < cols) {
      fprintf(stderr,
              "error: ldb (%d) must not be smaller than cols (%d)"
              " for row major, no trans type\n",
              ldb, cols);
      XERBLA(1);
    }
  } else {
    if (ldb < rows) {
      fprintf(stderr,
              "error: ldb (%d) must not be smaller than rows (%d)"
              " for row major, trans type\n",
              ldb, rows);
      XERBLA(1);
    }
  }

  if (trans == CblasTrans && (rows % 4 != 0 || cols % 4 != 0)) {
    fprintf(stderr,
            "error: rows (%d) and cols (%d) must be a multiple of four"
            " for trans for now\n",
            rows, cols);
    XERBLA(1);
  }

  uint32_t a_handle, b_handle;
  uint32_t a_bus, b_bus;

  qmkl6.locate_virt((void*)a, a_handle, a_bus);
  qmkl6.locate_virt((void*)b, b_handle, b_bus);

  qmkl6.unif[0] = (layout == CblasRowMajor) ? rows : cols;
  qmkl6.unif[1] = (layout == CblasRowMajor) ? cols : rows;
  qmkl6.unif[2] = qmkl6.bit_cast<uint32_t>(alpha);
  qmkl6.unif[3] = a_bus;
  qmkl6.unif[4] = 4 * lda;
  qmkl6.unif[5] = b_bus;
  qmkl6.unif[6] = 4 * ldb;

  qmkl6.execute_qpu_code(
      (trans == CblasNoTrans)
          ? qmkl6.qpu_somatcopy_n_bus
          : (qmkl6.unif[0] % 256 == 0 && qmkl6.unif[1] % 32 == 0)
                ? qmkl6.qpu_somatcopy_t_256x32_bus
                : qmkl6.qpu_somatcopy_t_4x4_bus,
      qmkl6.unif_bus, num_qpus, 1, b_handle);

  qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, b_handle);
}

void qmkl6_context::init_blaslike(void) {
  qpu_somatcopy_n =
      (uint64_t*)alloc_memory(sizeof(qpu_somatcopy_n_orig),
                              qpu_somatcopy_n_handle, qpu_somatcopy_n_bus);
  memcpy(qpu_somatcopy_n, qpu_somatcopy_n_orig, sizeof(qpu_somatcopy_n_orig));
  qpu_somatcopy_t_4x4 = (uint64_t*)alloc_memory(
      sizeof(qpu_somatcopy_t_4x4_orig), qpu_somatcopy_t_4x4_handle,
      qpu_somatcopy_t_4x4_bus);
  memcpy(qpu_somatcopy_t_4x4, qpu_somatcopy_t_4x4_orig,
         sizeof(qpu_somatcopy_t_4x4_orig));
  qpu_somatcopy_t_256x32 = (uint64_t*)alloc_memory(
      sizeof(qpu_somatcopy_t_256x32_orig), qpu_somatcopy_t_256x32_handle,
      qpu_somatcopy_t_256x32_bus);
  memcpy(qpu_somatcopy_t_256x32, qpu_somatcopy_t_256x32_orig,
         sizeof(qpu_somatcopy_t_256x32_orig));
}

void qmkl6_context::finalize_blaslike(void) {
  free_memory(sizeof(qpu_somatcopy_t_256x32_orig),
              qpu_somatcopy_t_256x32_handle, qpu_somatcopy_t_256x32);
  free_memory(sizeof(qpu_somatcopy_t_4x4_orig), qpu_somatcopy_t_4x4_handle,
              qpu_somatcopy_t_4x4);
  free_memory(sizeof(qpu_somatcopy_n_orig), qpu_somatcopy_n_handle,
              qpu_somatcopy_n);
}
