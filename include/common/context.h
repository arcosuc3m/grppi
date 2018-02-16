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
#ifndef GRPPI_COMMON_CONTEXT_H
#define GRPPI_COMMON_CONTEXT_H

#include <type_traits>

namespace grppi {

/**
\brief Representation of a context pattern.
Represents a context that uses a given policy to run a transformer.
This pattern is intended to switch between execution policies 
in a pattern composition.
\tparam ExecutionPolicy Execution policy type for the context.
\tparam Transformer Callable type for the context transformer.
*/
template <typename ExecutionPolicy, typename Transformer>
class context_t {
public:

  using transformer_type = Transformer;
  using execution_policy_type = ExecutionPolicy;

  /**
  \brief Constructs a context with a execution policy and a transformer.
  \param e Context to run the transformer function.
  \param t Transformer to be run.
  */
  context_t(ExecutionPolicy & e, Transformer && t) noexcept :
    execution_policy_{e}, transformer_{t}
  {}

  /**
  \brief Return the execution policy used in the context.
  \return The execution policy. 
  */
  ExecutionPolicy & execution_policy(){
    return execution_policy_;
  }
  

  /**
  \brief Return the transformer function.
  \return The transformer function. 
  */
  Transformer & transformer(){
    return transformer_;
  }

  /**
  \brief Invokes the trasnformer of the context over a data item.
  */
  template <typename I>
  auto operator()(I && item) const {
    return transformer_(std::forward<I>(item));
  }

private:
  Transformer transformer_;
  ExecutionPolicy& execution_policy_;
};

namespace internal {

template<typename T>
struct is_context : std::false_type {};

template<typename E, typename T>
struct is_context<context_t<E,T>> : std::true_type {};

} // namespace internal

template <typename T>
static constexpr bool is_context = internal::is_context<std::decay_t<T>>();

template <typename T>
using requires_context = typename std::enable_if_t<is_context<T>, int>;

template <typename T, typename U>
using concept_context = typename std::enable_if_t<is_context<T>, U>;

}

#endif
