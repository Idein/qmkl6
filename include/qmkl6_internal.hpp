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

        MKLExitHandler exit_handler = exit;

        struct memory_area {
            size_t alloc_size;
            uint32_t handle, bus_addr_aligned;
            void *virt_addr;
        };

        std::unordered_map <const void*, struct memory_area> memory_map;

        uint32_t unif_handle;
        uint32_t unif_bus;
        uint32_t *unif;

        /* qmkl6.cpp */

        qmkl6_context(void);
        ~qmkl6_context(void);

        void execute_qpu_code(uint32_t qpu_code_bus, uint32_t unif_bus,
                unsigned num_qpus, unsigned num_handles, ...);
        void wait_for_handles(uint64_t timeout_ns, unsigned num_handles, ...);
        void* alloc_memory(size_t size, uint32_t &handle, uint32_t &bus_addr);
        void free_memory(size_t size, uint32_t handle, void *map);
        void locate_virt(const void *virt_addr, uint32_t &handle,
                uint32_t &bus_addr);

    private:

        int drm_fd;

        size_t unif_size;

        /* support.cpp */

        void init_support(void);
        void finalize_support(void);
};

extern qmkl6_context qmkl6;


#endif /* _QMKL6_INTERNAL_HPP_ */
