# Doxygen generation
find_package(Doxygen)
if (DOXYGEN_FOUND)

    # Input file
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/doxy/Doxyfile)
    
    add_custom_target( doxy_generation
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )

    add_custom_target( doc_doxygen
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_map.html ${CMAKE_BINARY_DIR}/doc/html/map_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_divide-conquer.html ${CMAKE_BINARY_DIR}/doc/html/divide-conquer_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_farm.html ${CMAKE_BINARY_DIR}/doc/html/farm_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_map-reduce.html ${CMAKE_BINARY_DIR}/doc/html/map-reduce_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_pipeline.html ${CMAKE_BINARY_DIR}/doc/html/pipeline_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_reduce.html ${CMAKE_BINARY_DIR}/doc/html/reduce_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_stencil.html ${CMAKE_BINARY_DIR}/doc/html/stencil_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_stream-filter.html ${CMAKE_BINARY_DIR}/doc/html/stream-filter_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_stream-iteration.html ${CMAKE_BINARY_DIR}/doc/html/stream-iteration_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_stream-reduce.html ${CMAKE_BINARY_DIR}/doc/html/stream-reduce_8md.html
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_install-notes.html
${CMAKE_BINARY_DIR}/doc/html/install-notes_8md.html
      DEPENDS doxy_generation)

else (DOXYGEN_FOUND)
  message("Doxygen is required to generate the documentation")
endif (DOXYGEN_FOUND)
