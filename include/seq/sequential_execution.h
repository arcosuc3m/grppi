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

#ifndef GRPPI_SEQ_SEQUENTIAL_EXECUTION_H
#define GRPPI_SEQ_SEQUENTIAL_EXECUTION_H

#include <type_traits>

namespace grppi{

/**
\brief Sequential execution policy.
*/
class sequential_execution {

public:

  /// \brief Default constructor.
  sequential_execution() {}

  /**
  \brief Set number of grppi threads.
  \note Setting concurrency degree is ignored for sequential execution.
  */
  void set_concurrency_degree(int n) const noexcept {}

  /**
  \brief Get number of grppi trheads.
  \note Getting concurrency degree is always 1 for sequential execution.
  */
  int concurrency_degree() const noexcept { return 1; }

  /**
  \brief Enable ordering.
  \note Enabling ordering of sequential execution is always ignored.
  */
  void enable_ordering() const noexcept {}

  /**
  \brief Disable ordering.
  \note Disabling ordering of sequential execution is always ignored.
  */
  void disable_ordering() const noexcept {}

  /**
  \brief Is execution ordered.
  \note Sequential execution is always ordered.
  */
  bool is_ordered() const noexcept { return true; }
};

/// Determine if a type is a sequential execution policy.
template <typename E>
constexpr bool is_sequential_execution() {
  return std::is_same<E, sequential_execution>::value;
}

template <typename E>
constexpr bool is_supported();

template <>
constexpr bool is_supported<sequential_execution>() {
  return true;
}

} // end namespace grppi

#endif
