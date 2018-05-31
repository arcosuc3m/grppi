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

# Find HWLOC
# -------
#
# Find the include and library directories for HWLOC library
#
# Use:
#
#   find_package(HWLOC)
#
# Modifying the following environment variable directs where search is performed
#
#   - HWLOC_ROOT Root search directory for HWLOC
#
# This module sets the following variables:
#  * HWLOC_FOUND Set to TRUE if HWLOC was found or FALSE otherwise
#  * HWLOC_INCLUDE_DIR Directory where HWLOC includes where found
#  * HWLOC_LIB HWLOC library
#
set(HWLOC_FOUND FALSE)

set(HWLOC_SEARCH_DIR $ENV{HWLOC_ROOT})

find_path(HWLOC_INCLUDE_DIR
  NAMES hwloc.h
  PATHS ${HWLOC_SEARCH_DIR} 
  PATH_SUFFIXES include
)

find_library(HWLOC_LIB
  NAMES hwloc
  DOC "hwloc library"
  PATHS ${HWLOC_SEARCH_DIR} 
  PATH_SUFFIXES lib
)


if(HWLOC_INCLUDE_DIR AND HWLOC_LIB)
  set(HWLOC_FOUND TRUE)
endif()
