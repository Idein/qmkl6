#include <cstdio>

#include "qmkl6.h"
#include "qmkl6_internal.hpp"


static const uint64_t qpu_scopy_orig[] = {
#include "scopy.qhex6"
};


void cblas_scopy(const int n, const float * const x, const int incx,
        float * const y, const int incy)
{
    if (n <= 0 || n % (16 * 8 * 8) != 0) {
        fprintf(stderr, "error: n (%d) must be a multiple of %d for now\n",
                n, 16 * 8 * 8);
        XERBLA(1);
    }
    if (incx <= 0 || incy <= 0) {
        fprintf(stderr, "error: inc must be greater than zero for now\n");
        XERBLA(1);
    }

    uint32_t x_handle, y_handle, x_bus, y_bus;
    qmkl6.locate_virt((void*) x, x_handle, x_bus);
    qmkl6.locate_virt((void*) y, y_handle, y_bus);

    qmkl6.unif[0] = n;
    qmkl6.unif[1] = x_bus;
    qmkl6.unif[2] = incx;
    qmkl6.unif[3] = y_bus;
    qmkl6.unif[4] = incy;

    qmkl6.execute_qpu_code(qmkl6.qpu_scopy_bus, qmkl6.unif_bus, 8, 1, y_handle);
    qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, y_handle);
}

void qmkl6_context::init_blas1(void)
{
    qpu_scopy = (uint64_t*) alloc_memory(sizeof(qpu_scopy_orig),
            qpu_scopy_handle, qpu_scopy_bus);
    memcpy(qpu_scopy, qpu_scopy_orig, sizeof(qpu_scopy_orig));
}

void qmkl6_context::finalize_blas1(void)
{
    free_memory(sizeof(qpu_scopy_orig), qpu_scopy_handle, qpu_scopy);
}
