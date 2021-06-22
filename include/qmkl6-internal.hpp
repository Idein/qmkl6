#ifndef _QMKL6_INTERNAL_HPP_
#define _QMKL6_INTERNAL_HPP_

#include <cstring>
#include <map>

#define XERBLA(info)                        \
  do {                                      \
    const int v = (info);                   \
    xerbla(__func__, &v, strlen(__func__)); \
    __builtin_unreachable();                \
  } while (0)

class qmkl6_context {
 public:
  MKLExitHandler exit_handler = exit;

  struct memory_area {
    size_t alloc_size, alloc_size_aligned;
    uint32_t handle, bus_addr_aligned;
    void *virt_addr;
  };

  std::map<const void *, struct memory_area> memory_map;

  uint32_t unif_handle, qpu_sasum_handle, qpu_saxpy_handle, qpu_scopy_handle,
      qpu_sdot_handle, qpu_snrm2_handle, qpu_sscal_handle, qpu_sgemv_n_handle,
      qpu_sgemv_t_handle, qpu_stbmv_handle, qpu_ctbmv_handle,
      qpu_sgemm_rnn_handle, qpu_sgemm_rnt_handle, qpu_sgemm_rtn_handle,
      qpu_sgemm_rtt_handle, qpu_somatcopy_n_handle, qpu_somatcopy_t_4x4_handle,
      qpu_somatcopy_t_256x32_handle, qpu_comatcopy_n_handle,
      qpu_comatcopy_t_4x4_handle, qpu_comatcopy_t_128x32_handle,
      qpu_fft2_handle, qpu_fft4_forw_handle, qpu_fft4_back_handle,
      qpu_fft8_forw_handle, qpu_fft8_back_handle;
  uint32_t unif_bus, qpu_sasum_bus, qpu_saxpy_bus, qpu_scopy_bus, qpu_sdot_bus,
      qpu_snrm2_bus, qpu_sscal_bus, qpu_sgemv_n_bus, qpu_sgemv_t_bus,
      qpu_stbmv_bus, qpu_ctbmv_bus, qpu_sgemm_rnn_bus, qpu_sgemm_rnt_bus,
      qpu_sgemm_rtn_bus, qpu_sgemm_rtt_bus, qpu_somatcopy_n_bus,
      qpu_somatcopy_t_4x4_bus, qpu_somatcopy_t_256x32_bus, qpu_comatcopy_n_bus,
      qpu_comatcopy_t_4x4_bus, qpu_comatcopy_t_128x32_bus, qpu_fft2_bus,
      qpu_fft4_forw_bus, qpu_fft4_back_bus, qpu_fft8_forw_bus,
      qpu_fft8_back_bus;
  uint32_t *unif;
  uint64_t *qpu_saxpy, *qpu_sasum, *qpu_scopy, *qpu_sdot, *qpu_snrm2,
      *qpu_sscal, *qpu_sgemv_n, *qpu_sgemv_t, *qpu_stbmv, *qpu_ctbmv,
      *qpu_sgemm_rnn, *qpu_sgemm_rnt, *qpu_sgemm_rtn, *qpu_sgemm_rtt,
      *qpu_somatcopy_n, *qpu_somatcopy_t_4x4, *qpu_somatcopy_t_256x32,
      *qpu_comatcopy_n, *qpu_comatcopy_t_4x4, *qpu_comatcopy_t_128x32,
      *qpu_fft2, *qpu_fft4_forw, *qpu_fft4_back, *qpu_fft8_forw, *qpu_fft8_back;

  uint64_t timeout_ns = UINT64_C(10'000'000'000);

  /* qmkl6.cpp */

  qmkl6_context(void);
  ~qmkl6_context(void);

  void execute_qpu_code(uint32_t qpu_code_bus, uint32_t unif_bus,
                        unsigned num_qpus, unsigned num_handles, ...);
  void wait_for_handles(uint64_t timeout_ns, unsigned num_handles, ...);
  void *alloc_memory(size_t size, uint32_t &handle, uint32_t &bus_addr);
  void *alloc_memory(size_t size, uint32_t &handle, uint32_t &bus_addr,
                     uint64_t &mmap_offset);
  void free_memory(size_t size, uint32_t handle, void *map);
  void locate_virt(const void *virt_addr, uint32_t &handle, uint32_t &bus_addr);

  template <typename T, typename U>
  T bit_cast(const U u) {
    static_assert(sizeof(T) == sizeof(U), "Size of T and U must match");

    union {
      T t;
      U u;
    } s = {
        .u = u,
    };

    return s.t;
  }

 private:
  int drm_fd;

  size_t unif_size;

  /* support.cpp */

  void init_support(void);
  void finalize_support(void);

  /* blas1.cpp */

  void init_blas1(void);
  void finalize_blas1(void);

  /* blas2.cpp */

  void init_blas2(void);
  void finalize_blas2(void);

  /* blas3.cpp */

  void init_blas3(void);
  void finalize_blas3(void);

  /* blaslike.cpp */

  void init_blaslike(void);
  void finalize_blaslike(void);

  /* fft.cpp */

  void init_fft(void);
  void finalize_fft(void);
};

extern qmkl6_context qmkl6;

#endif /* _QMKL6_INTERNAL_HPP_ */
