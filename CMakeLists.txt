# For FindPython3.
cmake_minimum_required(VERSION 3.12)

project(qmkl6 VERSION 0.0.0 LANGUAGES CXX
        DESCRIPTION "Math Kernel Library for VideoCore VI QPU")

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_CONTACT "Yukimasa Sugizaki <ysugi@idein.jp>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_DEBIAN_PACKAGE_DEPENDS libdrm_v3d)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX /usr CACHE PATH "Install prefix" FORCE)
endif ()
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
message(STATUS "Install prefix is set to ${CMAKE_INSTALL_PREFIX}")

include(GNUInstallDirs)
include(CPack)
include(CTest)

find_package(PkgConfig)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

pkg_check_modules(DRM_V3D REQUIRED libdrm_v3d)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

configure_file(qmkl6.pc.in qmkl6.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qmkl6.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
