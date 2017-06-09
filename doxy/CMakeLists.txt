# Doxygen generation
find_package(Doxygen)
if (DOXYGEN_FOUND)

    # Input file
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/doxy/Doxyfile)
    
    add_custom_target( doc_doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )

else (DOXYGEN_FOUND)
  message("Doxygen is required to generate the documentation")
endif (DOXYGEN_FOUND)