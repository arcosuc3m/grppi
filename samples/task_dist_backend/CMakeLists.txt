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

add_executable(task_dist_test main.cpp )

target_include_directories(task_dist_test PUBLIC
  ${ZMQ_INCLUDE_DIR}
  ${BOOST_INCLUDEDIR}
)

target_link_libraries(task_dist_test 
  ${ZMQ_LIBRARY}
  ${Boost_LIBRARIES})
