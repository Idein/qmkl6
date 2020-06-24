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

#include "qmkl6.h"
#include "qmkl6_internal.hpp"

#include <drm_v3d.h>


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
}

qmkl6_context::~qmkl6_context(void)
{
    int ret;

    mkl_set_exit_handler(exit);

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
