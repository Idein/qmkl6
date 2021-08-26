#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "cblas-qmkl6.h"
#include "qmkl6-internal.hpp"

static const uint64_t qpu_fft2_orig[] = {
#include "fft2.qhex6"
};

static const uint64_t qpu_fft4_forw_orig[] = {
#include "fft4_forw.qhex6"
};

static const uint64_t qpu_fft4_back_orig[] = {
#include "fft4_back.qhex6"
};

static const uint64_t qpu_fft8_forw_orig[] = {
#include "fft8_forw.qhex6"
};

static const uint64_t qpu_fft8_back_orig[] = {
#include "fft8_back.qhex6"
};

struct fftwf_plan_s {
  int n, log2n, radix;
  bool is_swapped;
  int sign;
  unsigned flags;
  size_t twiddle_size;

  uint32_t unif_handle, twiddle_handle, in_handle, out_handle, temp_handle;
  uint32_t code_bus, unif_bus, twiddle_bus, in_bus, out_bus, temp_bus;
  uint32_t *unif;
  std::complex<float> *twiddle, *in, *out, *temp;
};

void *fftwf_malloc(const size_t n) { return mkl_malloc(n, 64); }

float *fftwf_alloc_real(const size_t n) {
  return (float *)fftwf_malloc(sizeof(float) * n);
}

fftwf_complex *fftwf_alloc_complex(const size_t n) {
  return (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * n);
}

void fftwf_free(void *const p) { mkl_free(p); }

static void append_omega(float **p, const float c, const int k, const int l,
                         const int m) {
  const std::complex<float> omega = std::polar(1.f, c * l * m / k);
  (*p)[0] = omega.real();
  (*p)[1] = omega.imag();
  *p += 2;
}

static void prepare_twiddle_2_or_4(void *const twiddle, const int n,
                                   const int radix, const int sign) {
  assert(radix == 2 || radix == 4);

  const float c = (float)sign * std::acos(-1.f) / (radix / 2);
  float *p = (float *)twiddle;
  for (int k = n / radix; k > 0; k /= radix) {
    for (int l = 0; l < k; ++l) {
      for (int m = 1; m < radix; ++m) append_omega(&p, c, k, l, m);
      if (radix >= 4)
        *p++ = qmkl6.bit_cast<float>(
            -int32_t(sizeof(float) * (2 * (radix - 1) + 1)));
    }
  }

  if (radix >= 4)
    assert(p ==
           (float *)twiddle + (n - 1) / (radix - 1) * (2 * (radix - 1) + 1));
  else
    assert(p == (float *)twiddle + (n - 1) * 2);
}

static void prepare_twiddle_8(void *const twiddle, const int n, const int radix,
                              const int sign) {
  assert(radix == 8);

  const float c = (float)sign * std::acos(-1.f) / (radix / 2),
              tmuc = qmkl6.bit_cast<float>(uint32_t(0xfafafafa));
  float *p = (float *)twiddle;
  for (int k = n / radix; k > 0; k /= radix) {
    for (int l = 0; l < k; ++l) {
      *p++ = tmuc;
      *p++ = tmuc;
      *p++ = tmuc;
      append_omega(&p, c, k, l, 1);
      append_omega(&p, c, k, l, 2);
      append_omega(&p, c, k, l, 3);
      append_omega(&p, c, k, l, 4);
      *p++ = tmuc;
      append_omega(&p, c, k, l, 5);
      append_omega(&p, c, k, l, 6);
      append_omega(&p, c, k, l, 7);
      *p++ = qmkl6.bit_cast<float>(
          -int32_t(sizeof(float) * (2 * (radix - 1) + 4 + 1)));
    }
  }

  assert(p ==
         (float *)twiddle + (n - 1) / (radix - 1) * (2 * (radix - 1) + 4 + 1));
}

fftwf_plan fftwf_plan_dft_1d(const int n, fftwf_complex *in, fftwf_complex *out,
                             const int sign,
                             [[maybe_unused]] const unsigned flags) {
  struct fftwf_plan_s *plan = new fftwf_plan_s;

  if (n <= 0) {
    fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
    XERBLA(1);
  } else if ((n & (n - 1)) != 0) {
    fprintf(stderr, "error: n (%d) must be a power of two for now\n", n);
    XERBLA(1);
  }
  plan->n = n;
  plan->log2n = std::log2(n);

  if (sign != FFTW_FORWARD && sign != FFTW_BACKWARD) {
    fprintf(stderr, "error: sign (%d) must be FFTW_FORWARD or FFTW_BACKWARD\n",
            sign);
    XERBLA(1);
  }
  plan->sign = sign;

  plan->in = (std::complex<float> *)in;
  plan->out = (std::complex<float> *)out;
  qmkl6.locate_virt((void *)in, plan->in_handle, plan->in_bus);
  qmkl6.locate_virt((void *)out, plan->out_handle, plan->out_bus);
  plan->temp = (std::complex<float> *)qmkl6.alloc_memory(
      sizeof(*plan->temp) * n, plan->temp_handle, plan->temp_bus);

  if (plan->log2n % 3 == 0 && n >= 64) {
    plan->radix = 8;
    plan->is_swapped = plan->log2n / 3 % 2;
    plan->code_bus = sign == FFTW_FORWARD ? qmkl6.qpu_fft8_forw_bus
                                          : qmkl6.qpu_fft8_back_bus;
    plan->twiddle_size = sizeof(float) * ((n - 1) / 7 * 19);
  } else if (plan->log2n % 2 == 0 && n >= 16) {
    plan->radix = 4;
    plan->is_swapped = plan->log2n / 2 % 2;
    plan->code_bus = sign == FFTW_FORWARD ? qmkl6.qpu_fft4_forw_bus
                                          : qmkl6.qpu_fft4_back_bus;
    plan->twiddle_size = sizeof(float) * ((n - 1) / 3 * 7);
  } else {
    plan->radix = 2;
    plan->is_swapped = plan->log2n % 2;
    plan->code_bus = qmkl6.qpu_fft2_bus;
    plan->twiddle_size = sizeof(plan->twiddle[0]) * (n - 1);
  }

  plan->twiddle = (std::complex<float> *)qmkl6.alloc_memory(
      plan->twiddle_size, plan->twiddle_handle, plan->twiddle_bus);
  if (plan->radix == 2 || plan->radix == 4)
    prepare_twiddle_2_or_4(plan->twiddle, n, plan->radix, sign);
  else
    prepare_twiddle_8(plan->twiddle, n, plan->radix, sign);

  plan->unif = (uint32_t *)qmkl6.alloc_memory(
      sizeof(*plan->unif) * 16, plan->unif_handle, plan->unif_bus);
  plan->unif[0] = n;
  /* unif[1..3] are set on execution. */
  if (plan->radix == 8) {
    plan->unif[4] = qmkl6.bit_cast<uint32_t>(-std::sqrt(2.f) / 2);
    plan->unif[5] = plan->twiddle_bus;
  } else
    plan->unif[4] = plan->twiddle_bus;

  return plan;
}

void fftwf_execute(const fftwf_plan plan) {
  plan->unif[1] = plan->in_bus;
  if (plan->is_swapped) {
    plan->unif[2] = plan->out_bus;
    plan->unif[3] = plan->temp_bus;
  } else {
    plan->unif[2] = plan->temp_bus;
    plan->unif[3] = plan->out_bus;
  }

  qmkl6.execute_qpu_code(plan->code_bus, plan->unif_bus, 8, 1,
                         plan->out_handle);
  qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, plan->out_handle);
}

void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex *const in,
                       fftwf_complex *const out) {
  uint32_t in_bus, out_bus;
  uint32_t in_handle, out_handle;
  qmkl6.locate_virt((void *)in, in_handle, in_bus);
  qmkl6.locate_virt((void *)out, out_handle, out_bus);

  plan->unif[1] = in_bus;
  if (plan->is_swapped) {
    plan->unif[2] = out_bus;
    plan->unif[3] = plan->temp_bus;
  } else {
    plan->unif[2] = plan->temp_bus;
    plan->unif[3] = out_bus;
  }

  qmkl6.execute_qpu_code(plan->code_bus, plan->unif_bus, 8, 1, out_handle);
  qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, out_handle);
}

void fftwf_destroy_plan(fftwf_plan plan) {
  qmkl6.free_memory(sizeof(*plan->unif) * 16, plan->unif_handle, plan->unif);
  qmkl6.free_memory(plan->twiddle_size, plan->twiddle_handle, plan->twiddle);
  qmkl6.free_memory(sizeof(*plan->temp) * plan->n, plan->temp_handle,
                    plan->temp);
  delete plan;
}

void qmkl6_context::init_fft(void) {
  qpu_fft2 = (uint64_t *)alloc_memory(sizeof(qpu_fft2_orig), qpu_fft2_handle,
                                      qpu_fft2_bus);
  qpu_fft4_forw = (uint64_t *)alloc_memory(
      sizeof(qpu_fft4_forw_orig), qpu_fft4_forw_handle, qpu_fft4_forw_bus);
  qpu_fft4_back = (uint64_t *)alloc_memory(
      sizeof(qpu_fft4_back_orig), qpu_fft4_back_handle, qpu_fft4_back_bus);
  qpu_fft8_forw = (uint64_t *)alloc_memory(
      sizeof(qpu_fft8_forw_orig), qpu_fft8_forw_handle, qpu_fft8_forw_bus);
  qpu_fft8_back = (uint64_t *)alloc_memory(
      sizeof(qpu_fft8_back_orig), qpu_fft8_back_handle, qpu_fft8_back_bus);
  memcpy(qpu_fft2, qpu_fft2_orig, sizeof(qpu_fft2_orig));
  memcpy(qpu_fft4_forw, qpu_fft4_forw_orig, sizeof(qpu_fft4_forw_orig));
  memcpy(qpu_fft4_back, qpu_fft4_back_orig, sizeof(qpu_fft4_back_orig));
  memcpy(qpu_fft8_forw, qpu_fft8_forw_orig, sizeof(qpu_fft8_forw_orig));
  memcpy(qpu_fft8_back, qpu_fft8_back_orig, sizeof(qpu_fft8_back_orig));
}

void qmkl6_context::finalize_fft(void) {
  free_memory(sizeof(qpu_fft8_back_orig), qpu_fft8_back_handle, qpu_fft8_back);
  free_memory(sizeof(qpu_fft8_forw_orig), qpu_fft8_forw_handle, qpu_fft8_forw);
  free_memory(sizeof(qpu_fft4_back_orig), qpu_fft4_back_handle, qpu_fft4_back);
  free_memory(sizeof(qpu_fft4_forw_orig), qpu_fft4_forw_handle, qpu_fft4_forw);
  free_memory(sizeof(qpu_fft2_orig), qpu_fft2_handle, qpu_fft2);
}
