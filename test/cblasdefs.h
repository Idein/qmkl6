#ifndef CBLASDEFS_H
#define CBLASDEFS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(CBLAS_atlas)
#include <cblas-atlas.h>
typedef CBLAS_ORDER CBLAS_LAYOUT;
#elif defined(CBLAS_netlib)
#include <cblas-netlib.h>
#elif defined(CBLAS_openblas)
#include <cblas-openblas.h>
#elif defined(CBLAS_mkl)
#include <mkl.h>
#elif defined(CBLAS_qmkl6)
#include <cblas-qmkl6.h>
#else
#error "CBLAS_* is not defined"
#endif

#if !defined(CBLAS_mkl) && !defined(CBLAS_qmkl6)

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__attribute__((__unused__)) static void* mkl_malloc(const size_t alloc_size,
                                                    const int alignment) {
  void* const p = aligned_alloc(alignment, alloc_size);
  if (p == NULL) {
    fprintf(stderr, "error: aligned_alloc: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  return p;
}

__attribute__((__unused__)) static void* mkl_calloc(const size_t num,
                                                    const size_t size,
                                                    const int alignment) {
  void* const p = mkl_malloc(num * size, alignment);
  memset(p, 0, num * size);
  return p;
}

__attribute__((__unused__)) static void mkl_free(void* const a_ptr) {
  free(a_ptr);
}

__attribute__((__unused__)) static double dsecnd(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

#endif /* !defined(CBLAS_mkl) && !defined(CBLAS_qmkl6) */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CBLASDEFS_H */
