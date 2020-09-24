#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <iostream>
#include <random>

#include "cblasdefs.h"

static int test_leak(void) {
  constexpr unsigned count = 1234;
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dist_size(1, 65536);
  std::uniform_int_distribution<unsigned> dist_align_shift(0, 16);
  std::array<void *, count> ptrs;

  unsigned AllocatedBuffers, AllocatedBuffers_init;
  uint64_t AllocatedBytes, AllocatedBytes_init;

  AllocatedBytes_init = mkl_mem_stat(&AllocatedBuffers_init);
  std::cout << "AllocatedBuffers (init): " << AllocatedBuffers_init
            << std::endl;
  std::cout << "AllocatedBytes (init): " << AllocatedBytes_init << " bytes"
            << std::endl;

  for (auto &ptr : ptrs) {
    const auto size = dist_size(gen);
    const auto align = 1 << dist_align_shift(gen);
    ptr = mkl_malloc(size, align);
  }

  AllocatedBytes = mkl_mem_stat(&AllocatedBuffers);
  std::cout << "AllocatedBuffers (max): " << AllocatedBuffers << std::endl;
  std::cout << "AllocatedBytes (max): " << AllocatedBytes << " bytes"
            << std::endl;

  std::shuffle(ptrs.begin(), ptrs.end(), gen);

  for (auto &ptr : ptrs) {
    mkl_free(ptr);
    ptr = NULL;
  }

  AllocatedBytes = mkl_mem_stat(&AllocatedBuffers);
  std::cout << "AllocatedBuffers (final): " << AllocatedBuffers << std::endl;
  std::cout << "AllocatedBytes (final): " << AllocatedBytes << " bytes"
            << std::endl;

  if (AllocatedBuffers != AllocatedBuffers_init ||
      AllocatedBytes != AllocatedBytes_init) {
    std::cerr << "error: Memory was leaked" << std::endl;
    return 1;
  }

  return 0;
}

static int test_retain(void) {
  constexpr size_t size = UINTMAX_C(0xc0ffee);
  uint64_t *ptr = (uint64_t *)mkl_calloc(size, sizeof(uint64_t), 4096);

  for (size_t i = 0; i < size; ++i) {
    if (ptr[i]) {
      std::cerr << "error: Memory area provided by calloc is not cleared"
                << std::endl;
      return 1;
    }
  }

  for (size_t i = 0; i < size; ++i) ptr[i] = i;

  asm volatile("" : "+m"(ptr) : : "memory");

  uint64_t sum = 0;
  for (size_t i = 0; i < size; ++i) sum += ptr[i];

  printf("Sum: %" PRIu64 "\n", sum);

  if (sum != (uint64_t)size * (size - 1) >> 1) {
    std::cerr << "error: The sum is different from expected" << std::endl;
    return 1;
  }

  return 0;
}

int main(void) {
  int ret;

  ret = test_leak();
  if (ret) return ret;

  ret = test_retain();
  if (ret) return ret;

  return 0;
}
