include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/include)
add_compile_options(-pipe -W -Wall -Wextra -O2 -g -std=c++17)

function (add_qhex6_from_py name)

    list(JOIN ARGN " " ARGN_string)

    add_custom_command(
        DEPENDS ${name}.py
        OUTPUT ${name}.qhex6
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
                ${ARGN} >${CMAKE_CURRENT_BINARY_DIR}/${name}.qhex6
        COMMENT "Assembling ${name}.py with arguments: ${ARGN_string}"
    )

endfunction ()

add_qhex6_from_py(scopy 8 0 0)
add_qhex6_from_py(sdot 8 4 0)

set_property(SOURCE blas1.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/scopy.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sdot.qhex6)

add_library(qmkl6 SHARED qmkl6.cpp support.cpp blas1.cpp)

install(TARGETS qmkl6 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
