# OMP
if(${OMP_ENABLE})
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()

set ( PATTERNS map )

foreach( pattern ${PATTERNS} )
  add_subdirectory(${pattern})
endforeach( pattern )
