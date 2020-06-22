
#ifndef _QMKL6_H_
#define _QMKL6_H_

#include <stdint.h>
#include <sys/types.h>


/* qmkl6.cpp */

__attribute__((constructor)) void qmkl6_init(void);
__attribute__((destructor)) void qmkl6_finalize(void);

/* support.cpp */

typedef void (*MKLExitHandler)(int why);

int mkl_set_exit_hander(MKLExitHandler myexit);
void xerbla(const char *srname, const int *info, int len);
double dsecond(void);
void* mkl_malloc(size_t alloc_size, int alignment);
void mkl_free(void *a_ptr);
uint64_t mkl_mem_stat(unsigned *AllocatedBuffers);


#endif /* _QMKL6_H_ */
