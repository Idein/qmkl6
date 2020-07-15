include_directories(${CMAKE_SOURCE_DIR}/include)
add_compile_options(-pipe -W -Wall -Wextra -O2 -g -std=c++17)
link_libraries(${DRM_V3D_LINK_LIBRARIES} qmkl6)

if (BUILD_TESTING)

    add_executable(mem mem.cpp)
    add_test(Mem mem)

    add_executable(saxpy saxpy.cpp)
    add_test(Saxpy saxpy)

    add_executable(scopy scopy.cpp)
    add_test(Scopy scopy)

    add_executable(sdot sdot.cpp)
    add_test(Sdot sdot)

endif ()
