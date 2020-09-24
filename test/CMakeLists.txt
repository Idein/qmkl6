# libatlas-base-dev, libblas-dev, libopenblas-dev
set(blas_list qmkl6 atlas netlib openblas)
set(test_list mem sasum saxpy scopy sdot snrm2 sscal sgemv sgemm)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options(-pipe -W -Wall -Wextra -O2 -g -std=c++17)

set(qmkl6_LINK_LIBRARIES ${DRM_V3D_LINK_LIBRARIES} qmkl6)

if (BUILD_TESTING)

    foreach (blas IN LISTS blas_list)

        if (NOT blas STREQUAL qmkl6)
            pkg_check_modules(${blas} blas-${blas})

            if (NOT ${blas}_FOUND)
                continue()
            endif ()
        endif ()

        foreach (test IN LISTS test_list)

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

endif ()
