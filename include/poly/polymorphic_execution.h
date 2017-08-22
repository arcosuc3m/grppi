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

#ifndef GRPPI_POLY_POLYMORPHIC_EXECUTION_H
#define GRPPI_POLY_POLYMORPHIC_EXECUTION_H

#include "../seq/sequential_execution.h"
#include "../native/parallel_execution_native.h"
#include "../tbb/parallel_execution_tbb.h"
#include "../omp/parallel_execution_omp.h"
#include "../ff/parallel_execution_ff.h"

#include <typeinfo>
#include <memory>

namespace grppi{

/// Meta-function to determine if a type is an execution policy
template <typename E>
constexpr bool is_execution_policy() {
  return is_sequential_execution<E>() 
      || is_parallel_execution_native<E>()
      || is_parallel_execution_tbb<E>()
      || is_parallel_execution_omp<E>()
	  || is_parallel_execution_ff<E>()
  ;
}

/// Simulate concept requirement for being an execution policy
template <typename E>
using requires_execution_policy =
  std::enable_if_t<is_execution_policy<E>(), int>;

/// Simulate concept requirement for not being an execution policy
template <typename E>
using requires_not_execution_policy =
  std::enable_if_t<!is_execution_policy<E>(), int>;

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

  // Check if the current execution is of type E
  template <typename E>
  bool is_execution() const {
    if (!has_execution()) return false;
    return typeid(E) == *execution_type_;
  }

  /// Get the execution pointer for a given type.
  template <typename E, 
            requires_execution_policy<E> = 0>
  E * execution_ptr() {
    if (!has_execution()) return nullptr;
    if (*execution_type_ != typeid(E)) return nullptr;
    return static_cast<E*>(execution_.get());
  }

  /// Get the execution pointer for a given type.
  template <typename E, 
            requires_not_execution_policy<E> = 0>
  E * execution_ptr() {
    return nullptr;
  }

private:
  /// Pointer to dynamically allocated execution policy.
  std::shared_ptr<void> execution_;

  /// Typeid of the current execution execution
  const std::type_info * execution_type_;

private:

  /// Create from other static policy with no arguments.
  /// @return A polymorphic execution containing an E object if E is supported
  /// @return An empty polymorphic execution otherwise.
  template <typename E>
  friend polymorphic_execution make_polymorphic_execution() {
    polymorphic_execution e;
    if (!is_supported<E>()) return e;
    e.execution_ = std::make_shared<E>();
    e.execution_type_ = &typeid(E);
    return e;
  }

};

} // end namespace grppi

#endif
