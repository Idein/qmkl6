#ifndef CXXUTILS_HPP
#define CXXUTILS_HPP

#include <random>
#include <string>
#include <type_traits>

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T, typename U = void>
struct uniform_distribution;

template <typename T>
struct uniform_distribution<T, std::enable_if_t<std::is_floating_point<T>{}>>
    : std::uniform_real_distribution<T> {};

template <typename T>
struct uniform_distribution<T, std::enable_if_t<is_complex<T>{}>> {
  using value_type = typename T::value_type;

 private:
  std::uniform_real_distribution<value_type> dist;

 public:
  uniform_distribution(void){};

  template <class Generator>
  T operator()(Generator& gen) {
    return T(dist(gen), dist(gen));
  }
};

template <typename T,
          typename std::enable_if_t<std::is_same<T, float>{}, bool> = true>
static constexpr const char* blas_prefix(void) {
  return "s";
};

template <typename T, typename std::enable_if_t<
                          std::is_same<T, std::complex<float>>{}, bool> = true>
static constexpr const char* blas_prefix(void) {
  return "c";
};

#endif /* CXXUTILS_HPP */
