
#ifndef _QMKL6_H_
#define _QMKL6_H_

#include <complex.h>
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

void cblas_ctbmv(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG diag, int n, int k, const void *a, int lda, void *x,
                 int incx);

/* blas3.cpp */

void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa,
                 CBLAS_TRANSPOSE transb, int m, int n, int k, float alpha,
                 const float *a, int lda, const float *b, int ldb, float beta,
                 float *c, int ldc);

/* blaslike.cpp */

void cblas_somatcopy(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int rows,
                     int cols, float alpha, const float *a, int lda, float *b,
                     int ldb);

void cblas_comatcopy(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int rows,
                     int cols, const void *alpha, const void *a, int lda,
                     void *b, int ldb);

/*
 * These definitions are derived from fftw3.h in FFTW 3.3.9.
 *
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)

#define FFTW_MEASURE (0U)
#define FFTW_DESTROY_INPUT (1U << 0)
#define FFTW_UNALIGNED (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE (1U << 3)     /* NO_EXHAUSTIVE is default */
#define FFTW_PRESERVE_INPUT (1U << 4) /* cancels FFTW_DESTROY_INPUT */
#define FFTW_PATIENT (1U << 5)        /* IMPATIENT is default */
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_WISDOM_ONLY (1U << 21)

/*
 * End of derivation from FFTW.
 */

/* fft.cpp */

typedef float _Complex fftwf_complex;
typedef struct fftwf_plan_s *fftwf_plan;

void *fftwf_malloc(size_t n);
float *fftwf_alloc_real(size_t n);
fftwf_complex *fftwf_alloc_complex(size_t n);
void fftwf_free(void *plan);

fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex *in, fftwf_complex *out,
                             int sign, unsigned flags);
void fftwf_execute(const fftwf_plan plan);
void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex *in,
                       fftwf_complex *out);
void fftwf_destroy_plan(fftwf_plan plan);

#if defined(__cplusplus)
}
#endif

#endif /* _QMKL6_H_ */
