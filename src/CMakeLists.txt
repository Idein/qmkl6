include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/include)
add_compile_options(-pipe -W -Wall -Wextra -O2 -g -std=c++17)

function (generate_qhex6_from_py name input)

    list(JOIN ARGN " " ARGN_string)

    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${name}.qhex6"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${input}"
                ${ARGN} > "${name}.qhex6"
        VERBATIM
        COMMENT "Generating ${name}.qhex6 from ${input} ${ARGN_string}"
    )

endfunction ()

generate_qhex6_from_py(sasum sasum.py 8 5 0)
generate_qhex6_from_py(saxpy saxpy.py 8 1 0)
generate_qhex6_from_py(scopy scopy.py 8 0 0)
generate_qhex6_from_py(sdot sdot.py 8 4 0)
generate_qhex6_from_py(snrm2 snrm2.py 8 5 0)
generate_qhex6_from_py(sscal sscal.py 8 0 0)

set_property(SOURCE blas1.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sasum.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/saxpy.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/scopy.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sdot.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/snrm2.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sscal.qhex6)

generate_qhex6_from_py(sgemv_n sgemv_n.py 8 0)
generate_qhex6_from_py(sgemv_t sgemv_t.py 8 0)
generate_qhex6_from_py(stbmv stbmv.py 8 1 0)

set_property(SOURCE blas2.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sgemv_n.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemv_t.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/stbmv.qhex6)

generate_qhex6_from_py(sgemm_rnn sgemm_rnn.py 8 0)
generate_qhex6_from_py(sgemm_rnt sgemm_rnt.py 8 0)
generate_qhex6_from_py(sgemm_rtn sgemm_rtn.py 8 0)
generate_qhex6_from_py(sgemm_rtt sgemm_rtt.py 8 0)

set_property(SOURCE blas3.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rnn.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rnt.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rtn.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/sgemm_rtt.qhex6)

generate_qhex6_from_py(somatcopy_n somatcopy_n.py 8 8 0)
generate_qhex6_from_py(somatcopy_t_4x4 somatcopy_t.py 8 1 1 4 4 0)
generate_qhex6_from_py(somatcopy_t_512x32 somatcopy_t.py 8 8 8 4 4 0)

set_property(SOURCE blaslike.cpp APPEND PROPERTY
        OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/somatcopy_n.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/somatcopy_t_4x4.qhex6
                       ${CMAKE_CURRENT_BINARY_DIR}/somatcopy_t_512x32.qhex6)

add_library(qmkl6 SHARED qmkl6.cpp support.cpp blas1.cpp blas2.cpp blas3.cpp
                         blaslike.cpp)

install(TARGETS qmkl6 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
