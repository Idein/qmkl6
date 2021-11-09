if (NOT BUILD_TESTING)
    return()
endif ()

find_package(OpenMP)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options(-pipe -W -Wall -Wextra -O2 -g -std=c++17)

# libblas-dev, libopenblas-dev, libatlas-base-dev
pkg_check_modules(netlib blas-netlib)
pkg_check_modules(openblas openblas)
pkg_check_modules(atlas blas-atlas)

# MKL tests are not built by default for users installed ICPC/ICPX without MKL.
# Set mkl_FOUND to TRUE to enable them.
set(mkl_CFLAGS -qmkl)
set(mkl_LINK_LIBRARIES -qmkl)

if (NOT NO_QMKL6)
    set(qmkl6_FOUND TRUE)
    set(qmkl6_LINK_LIBRARIES ${DRM_V3D_LINK_LIBRARIES} qmkl6)
endif (NOT NO_QMKL6)

foreach (blas IN ITEMS qmkl6 netlib openblas atlas mkl)

    if (NOT ${blas}_FOUND)
        continue()
    endif ()

    foreach (test IN ITEMS mem asum axpy copy dot nrm2 scal gemv tbmv gemm
                           omatcopy)

        if (test STREQUAL mem AND NOT blas STREQUAL qmkl6)
            continue()
        endif ()

        if (test STREQUAL omatcopy
            AND NOT blas MATCHES "^(openblas|mkl|qmkl6)$")
            continue()
        endif ()

        add_executable(${test}-${blas} ${test}.cpp)
        target_compile_definitions(${test}-${blas} PUBLIC CBLAS_${blas})
        target_compile_options(${test}-${blas} PUBLIC ${${blas}_CFLAGS})
        target_link_libraries(${test}-${blas} ${${blas}_LINK_LIBRARIES})
        if (OpenMP_CXX_FOUND)
            target_link_libraries(${test}-${blas} OpenMP::OpenMP_CXX)
        endif ()
        add_test(${test}-${blas} ${test}-${blas})

    endforeach()

endforeach ()

# libfftw3-dev
pkg_check_modules(fftw3 fftw3f)

foreach (fftw IN ITEMS qmkl6 fftw3 mkl)

    if (NOT ${fftw}_FOUND)
        continue()
    endif ()

    add_executable(fft-${fftw} fft.cpp)
    target_compile_definitions(fft-${fftw} PUBLIC FFT_${fftw})
    target_compile_options(fft-${fftw} PUBLIC ${${fftw}_CFLAGS})
    target_link_libraries(fft-${fftw} ${${fftw}_LINK_LIBRARIES})
    if (OpenMP_CXX_FOUND)
        target_link_libraries(fft-${fftw} OpenMP::OpenMP_CXX)
    endif ()
    add_test(fft-${fftw} fft-${fftw})

endforeach ()
