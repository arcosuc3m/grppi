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

#include_directories("${BOOST_INCLUDEDIR}")
#include_directories("${ZMQ_INCLUDE_DIR}")
#message( ${ZMQ_INCLUDE_DIR} )

add_executable(string2bin string2bin.cpp )
target_include_directories(string2bin PUBLIC
  ${ZMQ_INCLUDE_DIR}
  ${BOOST_INCLUDEDIR}
)
target_link_libraries(string2bin
  ${ZMQ_LIBRARY}
  ${Boost_LIBRARIES})


add_executable(bin2string bin2string.cpp )
target_include_directories(bin2string PUBLIC
  ${ZMQ_INCLUDE_DIR}
  ${BOOST_INCLUDEDIR}
)
target_link_libraries(bin2string
  ${ZMQ_LIBRARY}
  ${Boost_LIBRARIES})
