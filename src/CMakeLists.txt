include_directories(${CMAKE_SOURCE_DIR}/include)
add_compile_options(-pipe -W -Wall -Wextra -O2 -g -std=c++17)

add_library(qmkl6 SHARED qmkl6.cpp support.cpp)

install(TARGETS qmkl6 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
