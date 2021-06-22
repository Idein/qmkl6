#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>

#if defined(FFT_fftw3) || defined(FFT_mkl)
#include <fftw3.h>
#elif defined(FFT_qmkl6)
#include <cblas-qmkl6.h>
#else
#error "FFT_* is not defined"
#endif

static double getsec(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

template <typename T>
class fft_impl {
  using U = std::complex<T>;

 public:
  enum domain {
    DOMAIN_COMPLEX,
    DOMAIN_REAL,
  };

  enum direction {
    DIRECTION_FORWARD,
    DIRECTION_BACKWARD,
  };

  virtual void execute(U *out, const U *in) = 0;
};

template <typename T>
class fft_fftw : public fft_impl<T> {
  using U = std::complex<T>;

  fftwf_plan plan;

 public:
  fft_fftw(const enum fft_impl<T>::domain domain,
           const enum fft_impl<T>::direction direction, const unsigned n) {
    assert(domain == fft_impl<T>::DOMAIN_COMPLEX);

    fftwf_complex *in = fftwf_alloc_complex(n), *out = fftwf_alloc_complex(n);

    plan = fftwf_plan_dft_1d(n, in, out,
                             (direction == fft_impl<T>::DIRECTION_FORWARD)
                                 ? FFTW_FORWARD
                                 : FFTW_BACKWARD,
                             FFTW_ESTIMATE);

    fftwf_free(out);
    fftwf_free(in);
  }

  ~fft_fftw(void) { fftwf_destroy_plan(plan); }

  void execute(U *const out, const U *const in) {
    fftwf_execute_dft(plan, (fftwf_complex *)in, (fftwf_complex *)out);
  }
};

template <unsigned radix, typename T>
class fft_stockham : public fft_impl<T> {
  using U = std::complex<T>;
  using V = double;

  static constexpr bool has_single_bit(unsigned x) {
    return x != 0 && !(x & (x - 1));
  }

  static constexpr int countr_zero(unsigned x) { return __builtin_ctz(x); }

 public:
  const unsigned n;
  const bool is_forward, is_swapped;
  U *temp, *twiddle;

  fft_stockham(const enum fft_impl<T>::domain domain,
               const enum fft_impl<T>::direction direction, const unsigned n)
      : n(n),
        is_forward(direction == fft_impl<T>::DIRECTION_FORWARD),
        is_swapped(countr_zero(n) / countr_zero(radix) % 2) {
    assert(domain == fft_impl<T>::DOMAIN_COMPLEX);
    assert(has_single_bit(n));
    assert(countr_zero(n) % countr_zero(radix) == 0);

    temp = new U[n];

    twiddle = new U[n - 1];
    U *p = twiddle;
    const V c =
        (is_forward ? V(-2. / radix) : V(2. / radix)) * std::acos(V(-1));
    for (unsigned k = n / radix; k > 0; k /= radix)
      for (unsigned l = 0; l < k; ++l)
        for (unsigned s = 1; s < radix; ++s)
          *p++ = std::polar(T(1), T(c * l / k * s));
    assert(p == twiddle + (n - 1));
  }

 private:
  template <unsigned r>
  inline void butterfly(const U x[r], U y[r]);

  template <>
  inline void butterfly<2>(const U x[2], U y[2]) {
    y[0] = x[0] + x[1];
    y[1] = x[0] - x[1];
  }

  template <>
  inline void butterfly<4>(const U x[4], U y[4]) {
    if constexpr (0) {
      /* DIF */
      const U c0 = x[0] + x[2], c1 = x[1] + x[3], c2 = x[0] - x[2],
              t = x[1] - x[3],
              c3 = is_forward ? U{t.imag(), -t.real()} : U{-t.imag(), t.real()};
      const U d0 = c0 + c1, d1 = c0 - c1, d2 = c2 + c3, d3 = c2 - c3;
      y[0] = d0;
      y[1] = d2;
      y[2] = d1;
      y[3] = d3;
    } else {
      /* DIT */
      const U c0 = x[0], c1 = x[2], c2 = x[1], c3 = x[3];
      const U d0 = c0 + c1, d1 = c0 - c1, d2 = c2 + c3, d3 = c2 - c3;
      const U t =
          is_forward ? U{d3.imag(), -d3.real()} : U{-d3.imag(), d3.real()};
      y[0] = d0 + d2;
      y[1] = d1 + t;
      y[2] = d0 - d2;
      y[3] = d1 - t;
    }
  }

 public:
  void execute(U *const out, const U *const in) {
    U *p = twiddle, *X = (U *)in, *Y = is_swapped ? out : temp;

    for (unsigned j = 1, k = n / radix; k > 0; j *= radix, k /= radix) {
      for (unsigned l = 0; l < k; ++l) {
        U omega[radix - 1];
        for (unsigned s = 1; s < radix; ++s) omega[s - 1] = *p++;
        for (unsigned m = 0; m < j; ++m) {
          U x[radix], y[radix];
          for (unsigned s = 0; s < radix; ++s)
            x[s] = X[n / radix * s + j * l + m];
          butterfly<radix>(x, y);
          Y[radix * j * l + m] = y[0];
          for (unsigned s = 1; s < radix; ++s)
            Y[j * s + radix * j * l + m] = y[s] * omega[s - 1];
        }
      }
      std::swap(X, Y);
      if (j == 1) Y = is_swapped ? temp : out;
    }
  }
};

template <typename T, template <typename> class fft>
class fft_sixstep : public fft_impl<T> {
  using U = std::complex<T>;

  unsigned n0, n1;
  U *temp, *twiddle;
  fft<T> *fft0, *fft1;

  void omatcopy_t(U *const out, const U *const in, const unsigned rows,
                  const unsigned cols) {
    constexpr unsigned block_rows = 32, block_cols = 32;
    static U block[block_rows * block_cols];

    for (unsigned i = 0; i < rows; i += block_rows) {
      for (unsigned j = 0; j < cols; j += block_cols) {
        const unsigned bound_ii =
                           (rows < i + block_rows) ? (rows - i) : block_rows,
                       bound_jj =
                           (cols < j + block_cols) ? (cols - j) : block_cols,
                       ld = block_cols;
        for (unsigned ii = 0; ii < bound_ii; ++ii)
          for (unsigned jj = 0; jj < bound_jj; ++jj)
            block[ld * ii + jj] = in[cols * i + j + cols * ii + jj];
        for (unsigned jj = 0; jj < bound_jj; ++jj)
          for (unsigned ii = 0; ii < bound_ii; ++ii)
            out[rows * j + i + rows * jj + ii] = block[ld * ii + jj];
      }
    }
  }

  void omatcopy_t_tbmv(U *const out, const U *const in, const unsigned rows,
                       const unsigned cols, const U *const diag) {
    constexpr unsigned block_rows = 32, block_cols = 32;
    static U block[block_rows * block_cols];

    for (unsigned i = 0; i < rows; i += block_rows) {
      for (unsigned j = 0; j < cols; j += block_cols) {
        const unsigned bound_ii =
                           (rows < i + block_rows) ? (rows - i) : block_rows,
                       bound_jj =
                           (cols < j + block_cols) ? (cols - j) : block_cols,
                       ld = block_cols;
        for (unsigned ii = 0; ii < bound_ii; ++ii)
          for (unsigned jj = 0; jj < bound_jj; ++jj)
            block[ld * ii + jj] = diag[cols * i + j + cols * ii + jj] *
                                  in[cols * i + j + cols * ii + jj];
        for (unsigned jj = 0; jj < bound_jj; ++jj)
          for (unsigned ii = 0; ii < bound_ii; ++ii)
            out[rows * j + i + rows * jj + ii] = block[ld * ii + jj];
      }
    }
  }

 public:
  fft_sixstep(const enum fft_impl<T>::domain domain,
              const enum fft_impl<T>::direction direction, const unsigned n) {
    const unsigned s = std::log2(n);
    assert(n == (unsigned)1 << s);
    n0 = 1 << (s / 2);
    n1 = (s % 2) ? n0 * 2 : n0;
    assert(n0 * n1 == n);

    this->fft0 = new fft<T>(domain, direction, n0);
    this->fft1 = new fft<T>(domain, direction, n1);

    temp = new U[n];

    twiddle = new U[n];
    const U c =
        U{0, (direction == fft_impl<T>::DIRECTION_FORWARD) ? T(-2) : T(2)} *
        std::acos(T(-1));
    for (unsigned j = 0; j < n1; ++j)
      for (unsigned k = 0; k < n0; ++k)
        twiddle[n0 * j + k] = std::exp(c * (T)j * (T)k / (T)n);
  }

  void execute(U *const out, const U *const in) {
    omatcopy_t(out, in, n0, n1);

    for (unsigned j = 0; j < n1; ++j)
      fft0->execute(&temp[n0 * j], &out[n0 * j]);

    omatcopy_t_tbmv(out, temp, n1, n0, twiddle);

    for (unsigned j = 0; j < n0; ++j)
      fft1->execute(&temp[n1 * j], &out[n1 * j]);

    omatcopy_t(out, temp, n0, n1);
  }
};

/*
 * A six-step FFT algorithm that unifies the transposition and multi-row FFT
 * phases to utilize the cache memory [1].
 *
 * [1] D. Takahashi, "A Blocking Algorithm for FFT on Cache-Based Processors",
 * 2001, https://doi.org/10.1007/3-540-48228-8_58
 */

template <typename T, class fft>
class fft_sixstep_block : public fft_impl<T> {
  using U = std::complex<T>;

  unsigned n0, n1, block0_len, block1_len;
  static constexpr unsigned pad0 = 8, pad1 = 8, pad2 = 8;
  U *twiddle, *temp, *block0, *block1;
  fft *fft0, *fft1;

  static constexpr unsigned l1c_size = 32 * 1024, l2c_size = 1024 * 1024;

 public:
  fft_sixstep_block(const enum fft_impl<T>::domain domain,
                    const enum fft_impl<T>::direction direction,
                    const unsigned n) {
    const unsigned s = std::log2(n);
    assert(n == (unsigned)1 << s);
    n0 = 1 << (s / 2);
    n1 = (s % 2) ? n0 * 2 : n0;
    assert(n0 * n1 == n);

    block0_len = std::max(l2c_size / 8 / n1 / 3, unsigned(1));
    block1_len = std::max(l2c_size / 8 / n0 / 4, unsigned(1));

    this->fft0 = new fft(domain, direction, n0);
    this->fft1 = new fft(domain, direction, n1);

    twiddle = new U[n];
    const U c =
        U{0, (direction == fft_impl<T>::DIRECTION_FORWARD) ? T(-2) : T(2)} *
        std::acos(T(-1));
    for (unsigned i = 0; i < n0; ++i)
      for (unsigned j = 0; j < n1; ++j)
        twiddle[n1 * i + j] = std::exp(c * T(i) * T(j) / T(n));

    temp = new U[n];
    block0 =
        new U[std::max((n0 + pad0) * block1_len, (n1 + pad2) * block0_len)];
    block1 = new U[(n0 + pad1) * block1_len];
  }

  void execute(U *const out, const U *const in) {
    for (unsigned j = 0; j < n1; j += block1_len) {
      const unsigned bound_jj = (j + block1_len < n1) ? block1_len : (n1 - j);

      for (unsigned i = 0; i < n0; ++i)
        for (unsigned jj = 0; jj < bound_jj; ++jj)
          block0[(n0 + pad0) * jj + i] = in[j + n1 * i + jj];

      for (unsigned jj = 0; jj < bound_jj; ++jj)
        fft0->execute(&block1[(n0 + pad1) * jj], &block0[(n0 + pad0) * jj]);

      for (unsigned i = 0; i < n0; ++i)
        for (unsigned jj = 0; jj < bound_jj; ++jj)
          temp[j + n1 * i + jj] =
              block1[(n0 + pad1) * jj + i] * twiddle[j + n1 * i + jj];
    }

    for (unsigned i = 0; i < n0; i += block0_len) {
      const unsigned bound_ii = (i + block0_len < n0) ? block0_len : (n0 - i);

      for (unsigned ii = 0; ii < bound_ii; ++ii)
        fft1->execute(&block0[(n1 + pad2) * ii], &temp[n1 * i + n1 * ii]);

      for (unsigned j = 0; j < n1; ++j)
        for (unsigned ii = 0; ii < bound_ii; ++ii)
          out[i + n0 * j + ii] = block0[j + (n1 + pad2) * ii];
    }
  }
};

template <typename T>
static class fft_impl<T> *fft_auto(const enum fft_impl<T>::domain domain,
                                   const enum fft_impl<T>::direction direction,
                                   const unsigned n) {
  const unsigned s = std::log2(n);
  assert(n == (unsigned)1 << s);

  if (s % 2 == 0) {
    if (s / 2 % 2 == 0 && n >= 1048576)
      return new fft_sixstep_block<T, fft_stockham<4, T>>(domain, direction, n);
    else
      return new fft_stockham<4, T>(domain, direction, n);
  } else if (n <= 65536)
    return new fft_stockham<2, T>(domain, direction, n);
  else
    return new fft_sixstep_block<T, fft_stockham<2, T>>(domain, direction, n);
}

template <typename T, class Generator>
static int test_fft_c2c_single(const enum fft_impl<T>::domain domain,
                               const enum fft_impl<T>::direction direction,
                               const unsigned n, Generator &gen) {
  using U = std::complex<T>;

  std::uniform_real_distribution<T> dist;

  printf("%s %s: n = %7d: ",
         (domain == fft_impl<T>::DOMAIN_COMPLEX)
             ? "c2c"
             : (direction == fft_impl<T>::DIRECTION_FORWARD) ? "r2c" : "c2r",
         (direction == fft_impl<T>::DIRECTION_FORWARD) ? "forw" : "back", n);

  U *in0 = new U[n], *out0 = new U[n];
  U *in1 = (U *)fftwf_alloc_complex(n), *out1 = (U *)fftwf_alloc_complex(n);

  std::generate(in0, in0 + n, std::bind(dist, gen));
  std::copy(in0, in0 + n, in1);

  fft_impl<T> *naive = fft_auto<T>(domain, direction, n);
  fft_impl<T> *fftw = new fft_fftw<T>(domain, direction, n);

  const double start0 = getsec();
  naive->execute(out0, in0);
  const double t0 = getsec() - start0;

  const double start1 = getsec();
  fftw->execute(out1, in1);
  const double t1 = getsec() - start1;

  float err_abs_min = HUGE_VALF, err_abs_max = -HUGE_VALF;
  float err_rel_min = HUGE_VALF, err_rel_max = -HUGE_VALF;
  for (unsigned i = 0; i < n; ++i) {
    const float err_abs = std::abs(out0[i] - out1[i]);
    const float err_rel = std::abs(err_abs / out0[i]);
    err_abs_min = std::min(err_abs_min, err_abs);
    err_abs_max = std::max(err_abs_max, err_abs);
    err_rel_min = std::min(err_rel_min, err_rel);
    err_rel_max = std::max(err_rel_max, err_rel);
  }

  printf(
      "err_abs = [%f, %f], "
      "err_rel = [%f, %f], "
      "%f sec -> %f sec, "
      "%f Melem/s -> %f Melem/s\n",
      err_abs_min, err_abs_max, err_rel_min, err_rel_max, t0, t1, n / t0 * 1e-6,
      n / t1 * 1e-6);

  if (err_rel_max > n * 1e-6f) {
    std::cerr << "error: Relative error is too large" << std::endl;
    return 1;
  }

  delete[] in0;
  delete[] out0;
  fftwf_free(in1);
  fftwf_free(out1);
  return 0;
}

int main(void) {
  setbuf(stdout, NULL);

  std::default_random_engine gen;
  int ret;

  for (unsigned i = 1; i < 21; ++i) {
    for (const enum fft_impl<float>::domain domain :
         {fft_impl<float>::DOMAIN_COMPLEX}) {
      for (const enum fft_impl<float>::direction direction :
           {fft_impl<float>::DIRECTION_FORWARD,
            fft_impl<float>::DIRECTION_BACKWARD}) {
        ret = test_fft_c2c_single<float>(domain, direction, 1 << i, gen);
        if (ret) return ret;
      }
    }
  }

  return 0;
}
