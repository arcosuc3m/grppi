set(samples
  arithmetic_add
)

foreach( sample ${samples} )
  add_subdirectory(${sample})     
endforeach( sample )