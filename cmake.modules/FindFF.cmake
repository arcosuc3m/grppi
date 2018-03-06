# @version		GrPPI v0.3
# @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
# @license		GNU/GPL, see LICENSE.txt
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You have received a copy of the GNU General Public License in LICENSE.txt
# also available in <http://www.gnu.org/licenses/gpl.html>.
#
# See COPYRIGHT.txt for copyright notices and details.

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
