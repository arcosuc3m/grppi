# Copyright 2018 Universidad Carlos III de Madrid
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/doc/html/md_context.html
${CMAKE_BINARY_DIR}/doc/html/context_8md.html
      DEPENDS doxy_generation)

else (DOXYGEN_FOUND)
  message("Doxygen is required to generate the documentation")
endif (DOXYGEN_FOUND)
