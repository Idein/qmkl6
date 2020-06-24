#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <drm_v3d.h>

#include "qmkl6.h"
#include "qmkl6_internal.hpp"


qmkl6_context qmkl6;

qmkl6_context::qmkl6_context(void)
{
    drm_fd = open("/dev/dri/card0", O_RDWR);
    if (drm_fd == -1) {
        fprintf(stderr, "error: open: %s\n", strerror(errno));
        XERBLA(drm_fd);
    }

    unif_size = sizeof(uint32_t) * 1024;
    unif = (uint32_t*) alloc_memory(unif_size, unif_handle, unif_bus);

    init_support();
    init_blas1();
}

qmkl6_context::~qmkl6_context(void)
{
    int ret;

    mkl_set_exit_handler(exit);

    finalize_blas1();
    finalize_support();

    free_memory(unif_size, unif_handle, unif);

    ret = close(drm_fd);
    if (ret) {
        fprintf(stderr, "error: close: %s\n", strerror(errno));
        XERBLA(ret);
    }
    drm_fd = -1;
}

void* qmkl6_context::alloc_memory(const size_t size, uint32_t &handle,
        uint32_t &bus_addr)
{
    int ret;

    ret = drm_v3d_create_bo(drm_fd, size, 0, &handle, &bus_addr);
    if (ret) {
        fprintf(stderr, "error: drm_v3d_create_bo: %s\n", strerror(errno));
        XERBLA(ret);
    }

    uint64_t mmap_offset;
    ret = drm_v3d_mmap_bo(drm_fd, handle, 0, &mmap_offset);
    if (ret) {
        fprintf(stderr, "error: drm_v3d_mmap_bo: %s\n", strerror(errno));
        XERBLA(ret);
    }

    void * const map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
            drm_fd, mmap_offset);
    if (map == MAP_FAILED) {
        fprintf(stderr, "error: mmap: %s\n", strerror(errno));
        XERBLA(ret);
    }

    return map;
}

void qmkl6_context::free_memory(const size_t size, const uint32_t handle,
        void * const map)
{
    int ret;

    ret = munmap(map, size);
    if (ret) {
        fprintf(stderr, "error: munmap: %s\n", strerror(errno));
        XERBLA(ret);
    }

    ret = drm_gem_close(drm_fd, handle);
    if (ret) {
        fprintf(stderr, "error: drm_gem_close: %s\n", strerror(errno));
        XERBLA(ret);
    }
}

void qmkl6_context::locate_virt(const void * const virt_addr,
        uint32_t &handle, uint32_t &bus_addr)
{
    const auto area = memory_map.find(virt_addr);
    if (area == memory_map.end()) {
        fprintf(stderr, "error: Memory area starting at %p is not known\n",
                virt_addr);
        XERBLA(1);
    }

    handle = area->second.handle;
    bus_addr = area->second.bus_addr_aligned;
}

void qmkl6_context::execute_qpu_code(const uint32_t qpu_code_bus,
        const uint32_t unif_bus, const unsigned num_qpus,
        const unsigned num_handles, ...)
{
    const uint32_t cfg[7] = {
        DRM_V3D_SET_FIELD(16, CSD_CFG0_NUM_WGS_X)
                | DRM_V3D_SET_FIELD(0, CSD_CFG0_WG_X_OFFSET),
        DRM_V3D_SET_FIELD(1, CSD_CFG1_NUM_WGS_Y)
                | DRM_V3D_SET_FIELD(0, CSD_CFG1_WG_Y_OFFSET),
        DRM_V3D_SET_FIELD(1, CSD_CFG2_NUM_WGS_Z)
                | DRM_V3D_SET_FIELD(0, CSD_CFG2_WG_Z_OFFSET),
        DRM_V3D_SET_FIELD(0, CSD_CFG3_MAX_SG_ID)
                | DRM_V3D_SET_FIELD(16 - 1, CSD_CFG3_BATCHES_PER_SG_M1)
                | DRM_V3D_SET_FIELD(16, CSD_CFG3_WGS_PER_SG)
                | DRM_V3D_SET_FIELD(16, CSD_CFG3_WG_SIZE),
        /* Number of batches, minus 1 */
        num_qpus - 1,
        /* Shader address, pnan, singleseg, threading, like a shader record. */
        qpu_code_bus,
        /* Uniforms address (4 byte aligned) */
        unif_bus,
    }, coef[4] = {
        0, 0, 0, 0
    };
    uint32_t handles[num_handles];

    va_list ap;
    va_start(ap, num_handles);
    for (unsigned i = 0; i < num_handles; ++i)
        handles[i] = va_arg(ap, uint32_t);
    va_end(ap);

    const int ret = drm_v3d_submit_csd(drm_fd, cfg, coef, handles,
            num_handles, 0, 0);
    if (ret) {
        fprintf(stderr, "error: drm_v3d_submit_csd: %s\n", strerror(errno));
        XERBLA(ret);
    }
}

void qmkl6_context::wait_for_handles(const uint64_t timeout_ns,
        const unsigned num_handles, ...)
{
    va_list ap;
    va_start(ap, num_handles);
    for (unsigned i = 0; i < num_handles; ++i) {
        const uint32_t handle = va_arg(ap, uint32_t);
        const int ret = drm_v3d_wait_bo(drm_fd, handle, timeout_ns);
        if (ret) {
            fprintf(stderr, "error: drm_v3d_wait_bo: %s\n", strerror(errno));
            XERBLA(ret);
        }
    }
    va_end(ap);
}
