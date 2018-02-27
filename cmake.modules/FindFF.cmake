# Modifying the following variables directs where search is performed
#
#   - FF_INCLUDE_DIR Directory for FF includes

set(FF_FOUND FALSE)

set(FF_SEARCH_DIR $ENV{FF_ROOT})

find_path(FF_INCLUDE_DIRS ff/config.hpp
  HINTS ${FF_SEARCH_DIR}
  PATH_SUFFIXES include
)

if(FF_INCLUDE_DIRS)
  set(FF_FOUND TRUE)
endif()
