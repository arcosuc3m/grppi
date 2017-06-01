# Doxygen generation
find_package(Doxygen)
if (DOXYGEN_FOUND)

    # Configure input / output files
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${PROJECT_SOURCE_DIR}/docs/Doxyfile.out)
    
    # File configuration
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target( doc_doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )

else (DOXYGEN_FOUND)
  message("Doxygen is required to generate the documentation")
endif (DOXYGEN_FOUND)