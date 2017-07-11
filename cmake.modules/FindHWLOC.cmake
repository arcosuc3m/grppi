message("searching hwloc")
find_path(HWLOC_INCLUDE_DIR
    NAMES 
       hwloc.h
    PATHS
       ${HWLOC_ROOT}
)

find_library(HWLOC_LIBS
    NAMES
       hwloc
    PATHS
       ${HWLOC_ROOT}
)


message("HWLOC ROOT: " ${HWLOC_ROOT} )
message("HWLOC INCLUDES : ${HWLOC_INCLUDE_DIR}")
message("HWLOC LIBS : ${HWLOC_LIBS}")


if(HWLOC_INCLUDE_DIR AND HWLOC_LIBS)
    message("searching hwloc ---- FOUND" )
    set(HWLOC_FOUND 1)
endif()

