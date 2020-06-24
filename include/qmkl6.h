
#ifndef _QMKL6_H_
#define _QMKL6_H_

#include <stdint.h>
#include <sys/types.h>


#if defined(__cplusplus)
extern "C" {
#endif


/* qmkl6.cpp */

void qmkl6_init(void);
void qmkl6_finalize(void);

/* support.cpp */

typedef void (*MKLExitHandler)(int why);

int mkl_set_exit_handler(MKLExitHandler myexit);
void xerbla(const char *srname, const int *info, int len);
double dsecond(void);
void* mkl_malloc(size_t alloc_size, int alignment);
void* mkl_calloc(size_t num, size_t size, int alignment);
void mkl_free(void *a_ptr);
uint64_t mkl_mem_stat(unsigned *AllocatedBuffers);



#if defined(__cplusplus)
}
#endif

#endif /* _QMKL6_H_ */
