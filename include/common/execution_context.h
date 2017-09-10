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
#ifndef GRPPI_COMMON_EXECUTION_CONTEXT_H
#define GRPPI_COMMON_EXECUTION_CONTEXT_H

#include <type_traits>

namespace grppi {

/**
\brief Representation of an execution context.
*/
template <typename Execution, typename Pipeline,
          requires_pipeline<Pipeline> = 0>
class execution_context_t {
public:

  using execution_type = Execution;
  using pipeline_type = Pipeline;
  using transformers_type = typename Pipeline::transformers_type;

  execution_context_t(const Execution & e, Pipeline && pipe) noexcept:
    execution_{e},
    pipeline_{pipe}
  {
    std::cerr << "execution context size -> " 
        << std::tuple_size<transformers_type>() << "\n";
  }

  execution_context_t(const Execution & e, const Pipeline & pipe) noexcept:
    execution_{e},
    pipeline_{pipe}
  {
    std::cerr << "execution context size -> " 
        << std::tuple_size<transformers_type>() << "\n";
  }

  auto transformers() const noexcept {
    return pipeline_.transformers();
  }

private:
  const Execution & execution_;
  Pipeline pipeline_;
};

namespace internal {

template<typename T>
struct is_execution_context : std::false_type {};

template <typename E, typename Pipeline>
struct is_execution_context<execution_context_t<E, Pipeline>> :std::true_type {};

} // namespace internal

template <typename T>
static constexpr bool is_execution_context = internal::is_execution_context<std::decay_t<T>>();

template <typename T>
using requires_execution_context = typename std::enable_if_t<is_execution_context<T>, int>;

template <typename T>
using requires_not_execution_context = typename std::enable_if_t<!is_execution_context<T>, int>;

} // namespace grppi

#endif
