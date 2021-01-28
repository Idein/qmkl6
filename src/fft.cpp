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

struct fftwf_plan_s {
  int n, log2n;
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

static void prepare_twiddle_radix2(std::complex<float> *const twiddle,
                                   const int n, const int sign) {
  const float c = (float)sign * std::acos(-1.f);
  std::complex<float> *p = twiddle;
  for (unsigned k = n / 2; k > 0; k /= 2)
    for (unsigned l = 0; l < k; ++l) *p++ = std::polar(1.f, c * l / k);
  assert(p == twiddle + n - 1);
}

static void prepare_twiddle_radix4(void *const twiddle, const int n,
                                   const int sign) {
  const float c = (float)sign * std::acos(-1.f) / 2;
  float *p = (float *)twiddle;
  for (unsigned k = n / 4; k > 0; k /= 4) {
    for (unsigned l = 0; l < k; ++l) {
      const std::complex<float> omega1 = std::polar(1.f, c * l / k),
                                omega2 = std::polar(1.f, c * l * 2 / k),
                                omega3 = std::polar(1.f, c * l * 3 / k);
      *p++ = omega1.real();
      *p++ = omega1.imag();
      *p++ = omega2.real();
      *p++ = omega2.imag();
      *p++ = omega3.real();
      *p++ = omega3.imag();
      *p++ = qmkl6.bit_cast<float>(int32_t(sizeof(float) * -7));
    }
  }
  assert(p == (float *)twiddle + (n - 1) / 3 * 7);
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

  if (plan->log2n % 2 == 0 && n >= 16) {
    plan->code_bus = sign == FFTW_FORWARD ? qmkl6.qpu_fft4_forw_bus
                                          : qmkl6.qpu_fft4_back_bus;
    plan->twiddle_size = sizeof(float) * ((n - 1) / 3 * 7);
    plan->twiddle = (std::complex<float> *)qmkl6.alloc_memory(
        plan->twiddle_size, plan->twiddle_handle, plan->twiddle_bus);
    prepare_twiddle_radix4(plan->twiddle, n, sign);
    plan->is_swapped = plan->log2n / 2 % 2;
  } else {
    plan->code_bus = qmkl6.qpu_fft2_bus;
    plan->twiddle_size = sizeof(plan->twiddle[0]) * (n - 1);
    plan->twiddle = (std::complex<float> *)qmkl6.alloc_memory(
        plan->twiddle_size, plan->twiddle_handle, plan->twiddle_bus);
    prepare_twiddle_radix2(plan->twiddle, n, sign);
    plan->is_swapped = plan->log2n % 2;
  }

  plan->unif = (uint32_t *)qmkl6.alloc_memory(
      sizeof(*plan->unif) * 16, plan->unif_handle, plan->unif_bus);
  plan->unif[0] = n;
  /* unif[1..3] are set on execution. */
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
  memcpy(qpu_fft2, qpu_fft2_orig, sizeof(qpu_fft2_orig));
  memcpy(qpu_fft4_forw, qpu_fft4_forw_orig, sizeof(qpu_fft4_forw_orig));
  memcpy(qpu_fft4_back, qpu_fft4_back_orig, sizeof(qpu_fft4_back_orig));
}

void qmkl6_context::finalize_fft(void) {
  free_memory(sizeof(qpu_fft4_back_orig), qpu_fft4_back_handle, qpu_fft4_back);
  free_memory(sizeof(qpu_fft4_forw_orig), qpu_fft4_forw_handle, qpu_fft4_forw);
  free_memory(sizeof(qpu_fft2_orig), qpu_fft2_handle, qpu_fft2);
}
