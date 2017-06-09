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

#ifndef GRPPI_POLYMORPHIC_EXECUTION_H
#define GRPPI_POLYMORPHIC_EXECUTION_H

#include <typeinfo>
#include <memory>

#include "common/seq_policy.h"
#include "common/thread_policy.h"
#include "common/tbb_policy.h"
#include "common/omp_policy.h"

namespace grppi{

/// Meta-function to determine if a type is an execution policy
template <typename E>
constexpr bool is_execution_policy() {
  return std::is_same<E, grppi::sequential_execution>::value 
      || std::is_same<E, grppi::parallel_execution_thr>::value
      || std::is_same<E, grppi::parallel_execution_tbb>::value
      || std::is_same<E, grppi::parallel_execution_omp>::value
  ;
}

/// Simulate concept requirement for an execution policy
template <typename E>
using requires_execution_policy =
  std::enable_if_t<is_execution_policy<E>(), int>;

// Forward declare polymorphic execution
class polymorphic_execution;

// Forward declare make execution for polymorphic execution
template <typename E>
polymorphic_execution make_polymorphic_execution();

/// Polymorphic execution supporting any grppi execution.
/// A polymorphic execution may hold either an execution or be empty.
class polymorphic_execution {
public:

  /// Create empty polymorphic execution.
  polymorphic_execution() noexcept :
    execution_{},
    execution_type_{nullptr}
  {}

  /// Determine if there is an execution stored.
  bool has_execution() const noexcept { return execution_.get() != nullptr; }

  /// Get the typeid of the current execution
  /// @pre has_execution()
  const std::type_info & type() const noexcept { 
    return *execution_type_; 
  }

  /// Get the execution pointer for a given type.
  template <typename E>
  E * execution_ptr() {
    return (*execution_type_ != typeid(E))?nullptr:static_cast<E*>(execution_.get());
  }

private:
  /// Pointer to dynamically allocated execution policy.
  std::shared_ptr<void> execution_;

  /// Typeid of the current execution execution
  const std::type_info * execution_type_;

private:

  /// Create from other static policy with no arguments.
  template <typename E>
  friend polymorphic_execution make_polymorphic_execution() {
    polymorphic_execution e;
    e.execution_ = std::make_shared<E>();
    e.execution_type_ = &typeid(E);
    return e;
  }

};


} // end namespace grppi

#endif
