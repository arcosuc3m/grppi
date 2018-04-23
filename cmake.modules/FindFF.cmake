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

# Find FF
# -------
#
# Find the include directories for FastFlow library
#
# Use:
#
#   find_package(FF)
#
# Modifying the following variables directs where search is performed
#
#   - FF_INCLUDE_DIR Directory for FF includes
#
# This module sets the following variables:
#  * FF_FOUND Set to TRUE if FastFlow was found or FALSE otherwise
#  * FF_INCLUDE_DIRS Directory where FastFlow includes where found
#
set(FF_FOUND FALSE)

set(FF_SEARCH_DIR $ENV{FF_ROOT})

find_path(FF_INCLUDE_DIRS ff/config.hpp
  HINTS ${FF_SEARCH_DIR}
  PATH_SUFFIXES include
)

if(FF_INCLUDE_DIRS)
  set(FF_FOUND TRUE)
endif()
