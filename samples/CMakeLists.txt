
include_directories( ${PROJECT_SOURCE_DIR} )

set(SAMPLES_map
  arithmetic_add
)

set ( PATTERNS map )

# OMP
if(${OMP_ENABLE})
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()

foreach( pattern ${PATTERNS} )

  foreach( sample ${SAMPLES_${PATTERNS}} )

    add_executable( ${pattern}_${sample} ${pattern}/${sample}/main.cpp )

    target_link_libraries( ${pattern}_${sample} pthread )

    target_link_libraries( ${pattern}_${sample} tbb )

    target_link_libraries( ${pattern}_${sample} ${Boost_LIBRARIES} )
          
  endforeach( sample )

endforeach( pattern )