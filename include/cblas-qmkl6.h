
#ifndef _QMKL6_H_
#define _QMKL6_H_

#include <stdint.h>
#include <sys/types.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  CblasRowMajor = 101,
  CblasColMajor = 102,
} CBLAS_LAYOUT;

typedef enum {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113,
} CBLAS_TRANSPOSE;

typedef enum {
  CblasUpper = 121,
  CblasLower = 122,
} CBLAS_UPLO;

typedef enum {
  CblasNonUnit = 131,
  CblasUnit = 132,
} CBLAS_DIAG;

/* support.cpp */

typedef void (*MKLExitHandler)(int why);

int mkl_set_exit_handler(MKLExitHandler myexit);
void xerbla(const char *srname, const int *info, int len);
double dsecnd(void);
void *mkl_malloc(size_t alloc_size, int alignment);
void *mkl_calloc(size_t num, size_t size, int alignment);
void mkl_free(void *a_ptr);
uint64_t mkl_mem_stat(unsigned *AllocatedBuffers);

/* blas1.cpp */

float cblas_sasum(int n, const float *x, int incx);
void cblas_saxpy(int n, float a, const float *x, int incx, float *y, int incy);
void cblas_scopy(int n, const float *x, int incx, float *y, int incy);
float cblas_sdot(int n, const float *x, int incx, const float *y, int incy);
float cblas_snrm2(int n, const float *x, int incx);
void cblas_sscal(int n, float a, float *x, int incx);

/* blas2.cpp */

void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
                 float alpha, const float *a, int lda, const float *x, int incx,
                 float beta, float *y, int incy);

void cblas_stbmv(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG diag, int n, int k, const float *a, int lda,
                 float *x, int incx);

/* blas3.cpp */

void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa,
                 CBLAS_TRANSPOSE transb, int m, int n, int k, float alpha,
                 const float *a, int lda, const float *b, int ldb, float beta,
                 float *c, int ldc);

#if defined(__cplusplus)
}
#endif

#endif /* _QMKL6_H_ */
