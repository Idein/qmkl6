#ifndef _QMKL6_INTERNAL_H_
#define _QMKL6_INTERNAL_H_

#include <string.h>


#define XERBLA(info) \
    do { \
        const int v = (info); \
        xerbla(__func__, &v, strlen(__func__)); \
        __builtin_unreachable(); \
    } while (0)


/* support.cpp */

uint32_t locate_bus_addr(void *virt_addr);

void qmkl6_init_support(void);
void qmkl6_finalize_support(void);


#endif /* _QMKL6_INTERNAL_H_ */
