#ifndef _QMKL6_INTERNAL_HPP_
#define _QMKL6_INTERNAL_HPP_

#include <cstring>
#include <unordered_map>


#define XERBLA(info) \
    do { \
        const int v = (info); \
        xerbla(__func__, &v, strlen(__func__)); \
        __builtin_unreachable(); \
    } while (0)


class qmkl6_context {

    public:

        /* qmkl6.cpp */

        MKLExitHandler exit_handler = exit;
        uint32_t *unif;

        qmkl6_context(void);
        ~qmkl6_context(void);

        /* support.cpp */

        struct memory_area {
            size_t alloc_size;
            uint32_t handle, bus_addr_aligned;
            void *virt_addr;
        };

        std::unordered_map <const void*, struct memory_area> memory_map;

        void* alloc_memory(size_t size, uint32_t &handle, uint32_t &bus_addr);
        void free_memory(size_t size, uint32_t handle, void *map);
        uint32_t locate_bus_addr(const void *virt_addr);

    private:

        /* support.cpp */

        int drm_fd;
        uint32_t unif_handle, unif_bus;

        void init_support(void);
        void finalize_support(void);
};

extern qmkl6_context qmkl6;


#endif /* _QMKL6_INTERNAL_HPP_ */
