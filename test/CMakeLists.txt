if (NOT BUILD_TESTING)
    return()
endif ()

find_package(OpenMP)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options(-pipe -W -Wall -Wextra -O2 -g -std=c++17)

# libblas-dev, libopenblas-dev, libatlas-base-dev
pkg_check_modules(netlib blas-netlib)
pkg_check_modules(openblas blas-openblas)
pkg_check_modules(atlas blas-atlas)

if (NOT NO_QMKL6)
    set(qmkl6_FOUND TRUE)
    set(qmkl6_LINK_LIBRARIES ${DRM_V3D_LINK_LIBRARIES} qmkl6)
endif (NOT NO_QMKL6)

foreach (blas IN ITEMS qmkl6 netlib openblas atlas)

    if (NOT ${blas}_FOUND)
        continue()
    endif ()

    foreach (test IN ITEMS mem sasum saxpy scopy sdot snrm2 sscal sgemv stbmv
                           sgemm)

        if (NOT blas STREQUAL qmkl6 AND test STREQUAL mem)
            continue()
        endif ()

        add_executable(${test}-${blas} ${test}.cpp)
        target_compile_definitions(${test}-${blas} PUBLIC CBLAS_${blas})
        target_link_libraries(${test}-${blas} ${${blas}_LINK_LIBRARIES})
        if (OpenMP_CXX_FOUND)
            target_link_libraries(${test}-${blas} OpenMP::OpenMP_CXX)
        endif ()
        add_test(${test}-${blas} ${test}-${blas})

    endforeach()

endforeach ()
