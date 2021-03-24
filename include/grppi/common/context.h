/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
  Transformer & transformer() & {
    return transformer_;
  }

  Transformer && transformer() && {
    return transformer_;
  }

  /**
  \brief Invokes the transformer of the context over a data item.
  */
  template <typename I>
  auto operator()(I && item) const {
    return transformer_(std::forward<I>(item));
  }

private:
  ExecutionPolicy& execution_policy_;
  Transformer transformer_;
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

}

#endif
