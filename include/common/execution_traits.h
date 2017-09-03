/**
* @version		GrPPI v0.3
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

#ifndef GRPPI_COMMON_EXECUTION_TRAITS_H
#define GRPPI_COMMON_EXECUTION_TRAITS_H

#include <type_traits>

namespace grppi {

/**
\brief Determines if an execution policy is supported in the current compilation.
\note This must be specialized by every execution policy.
*/
template <typename E>
constexpr bool is_supported() { return false; }

/**
\brief Determines if an execution policy supports the map pattern.
\note This must be specialized by every execution policy supporting map.
*/
template <typename E>
constexpr bool supports_map() { return false; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note This must be specialized by every execution policy supporting reduce.
*/
template <typename E>
constexpr bool supports_reduce() { return false; }

} // end namespace grppi

#endif
