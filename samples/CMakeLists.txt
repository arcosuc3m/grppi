# Data-parallel patterns
add_subdirectory(map)
add_subdirectory(reduce)
add_subdirectory(map-reduce)
add_subdirectory(stencil)

# Task-parallel patterns
add_subdirectory(divide-conquer)

# Streaming patterns
add_subdirectory(pipeline)
add_subdirectory(farm)
add_subdirectory(stream-reduce)
add_subdirectory(stream-filter)
add_subdirectory(stream-iteration)

add_subdirectory(windowed_farm)
