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

#ifndef GRPPI_TBB_POLICY_H
#define GRPPI_TBB_POLICY_H

// Only if compiled with TBB enabled
#ifdef GRPPI_TBB

#include <type_traits>

namespace grppi{

/** @brief Set the execution mode to parallel with threading building blocks
 *    (TBB) framework implementation
 */
struct parallel_execution_tbb{
  bool ordering = true;
  bool lockfree = false;
  int num_threads = 4;
  int num_tokens = 100;
  /** @brief Set num_threads to the maximum number of thread available by the
   *    hardware
   */
  parallel_execution_tbb(){};

  /** @brief Set num_threads to _threads in order to run in parallel
   *
   *  @param _threads number of threads used in the parallel mode
   */
  parallel_execution_tbb(int _threads){ num_threads= _threads; };
};

template <typename E>
constexpr bool is_parallel_execution_tbb() {
  return std::is_same<E, parallel_execution_tbb>::value;
}

template <typename E>
constexpr bool is_supported();

template <>
constexpr bool is_supported<parallel_execution_tbb>() {
  return true;
}


} // end namespace grppi

#else // GRPPI_TBB not defined

namespace grppi {

/// Parallel execution policy.
/// Empty type if GRPPI_TBB disabled.
struct parallel_execution_tbb {};

/// Determine if a type is a TBB execution policy.
/// False if GRPPI_TBB disabled.
template <typename E>
constexpr bool is_parallel_execution_tbb() {
  return false;
}

template <typename E>
constexpr bool is_supported();

template <>
constexpr bool is_supported<parallel_execution_tbb>() {
  return false;
}

}

#endif // GRPPI_TBB

#endif
