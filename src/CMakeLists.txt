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

add_qhex6_from_py(sasum 8 5 0)
add_qhex6_from_py(saxpy 8 1 0)
add_qhex6_from_py(scopy 8 0 0)
add_qhex6_from_py(sdot 8 4 0)
add_qhex6_from_py(snrm2 8 5 0)
add_qhex6_from_py(sscal 8 0 0)

set_property(SOURCE blas1.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sasum.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/saxpy.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/scopy.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sdot.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/snrm2.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sscal.qhex6)

add_qhex6_from_py(sgemv_n 8 0)
add_qhex6_from_py(sgemv_t 8 0)
add_qhex6_from_py(stbmv 8 1 0)

set_property(SOURCE blas2.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sgemv_n.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemv_t.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/stbmv.qhex6)

add_qhex6_from_py(sgemm_rnn 8 0)
add_qhex6_from_py(sgemm_rnt 8 0)
add_qhex6_from_py(sgemm_rtn 8 0)
add_qhex6_from_py(sgemm_rtt 8 0)

set_property(SOURCE blas3.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rnn.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rnt.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rtn.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rtt.qhex6)

add_library(qmkl6 SHARED qmkl6.cpp support.cpp blas1.cpp blas2.cpp blas3.cpp)

install(TARGETS qmkl6 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
