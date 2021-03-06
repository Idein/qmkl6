#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unordered_map>

#include "cblas-qmkl6.h"
#include "qmkl6-internal.hpp"

int mkl_set_exit_handler(const MKLExitHandler myexit) {
  qmkl6.exit_handler = (myexit == NULL) ? exit : myexit;
  return 0;
}

void xerbla(const char* const srname, const int* const info,
            [[maybe_unused]] const int len) {
  fprintf(stderr, "QMKL6 error: %s: %d\n", srname, *info);
  qmkl6.exit_handler(EXIT_FAILURE);
}

double dsecnd(void) {
  struct timespec t;

  clock_gettime(CLOCK_MONOTONIC, &t);

  return t.tv_sec + t.tv_nsec * 1e-9;
}

void* mkl_malloc(const size_t alloc_size, int alignment) {
  if (alignment <= 0 || alignment & (alignment - 1)) alignment = 32;

  const size_t alloc_size_aligned = alloc_size + alignment - 1;

  uint32_t handle, bus_addr;
  void* const virt_addr =
      qmkl6.alloc_memory(alloc_size_aligned, handle, bus_addr);

  /* bus_addr + offset ≡ 0 (mod alignment) */
  const uint32_t offset = -bus_addr & (alignment - 1);
  void* const virt_addr_aligned = (void*)((uintptr_t)virt_addr + offset);

  struct qmkl6_context::memory_area area = {
      .alloc_size = alloc_size,
      .alloc_size_aligned = alloc_size_aligned,
      .handle = handle,
      .bus_addr_aligned = bus_addr + offset,
      .virt_addr = virt_addr,
  };

  qmkl6.memory_map.emplace(virt_addr_aligned, area);

  return virt_addr_aligned;
}

void* mkl_calloc(const size_t num, const size_t size, const int alignment) {
  void* const virt = mkl_malloc(size * num, alignment);
  memset(virt, 0, size * num);
  return virt;
}

void mkl_free(void* const a_ptr) {
  if (a_ptr == NULL) return;

  const auto area = qmkl6.memory_map.find(a_ptr);
  if (area == qmkl6.memory_map.end()) {
    fprintf(stderr, "error: Memory area starting at %p is not known\n", a_ptr);
    XERBLA(1);
  }

  qmkl6.free_memory(area->second.alloc_size_aligned, area->second.handle,
                    area->second.virt_addr);

  qmkl6.memory_map.erase(area);
}

uint64_t mkl_mem_stat(unsigned* AllocatedBuffers) {
  *AllocatedBuffers = qmkl6.memory_map.size();

  uint64_t AllocatedBytes = 0;
  for (auto& mem : qmkl6.memory_map)
    AllocatedBytes += mem.second.alloc_size_aligned;
  return AllocatedBytes;
}

void qmkl6_context::init_support(void) {}

void qmkl6_context::finalize_support(void) {}
