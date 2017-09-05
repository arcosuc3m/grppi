/**
* @version		GrPPI v0.2
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef GRPPI_COMMON_PATTERNS_H
#define GRPPI_COMMON_PATTERNS_H

#include <tuple> 

#include "farm_pattern.h"
#include "filter_pattern.h"
#include "pipeline_pattern.h"
#include "reduce_pattern.h"
#include "iteration_pattern.h"

namespace grppi{

template <typename T>
constexpr bool is_no_pattern =
  !is_farm<T> && 
  !is_filter<T> && 
  !is_pipeline<T> &&
  !is_reduce<T> &&
  !is_iteration<T>;

template <typename T>
using requires_no_pattern = std::enable_if_t<is_no_pattern<T>,int>;

} // end namespace grppi

#endif
