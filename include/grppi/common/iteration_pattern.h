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
#ifndef GRPPI_COMMON_ITERATION_PATTERN_H
#define GRPPI_COMMON_ITERATION_PATTERN_H

#include <type_traits>
#include "meta.h"

namespace grppi {

/**
\brief Representation of iteration pattern.
Represents a iteration that can be used as a stage on a pipeline.
\tparam Predicate Callable type for the iteration predicate.
*/
template <typename Transformer, typename Predicate>
class iteration_t {
public:

  using input_type = meta::input_type<Transformer>;
  using output_type = meta::output_type<Transformer>;

  /**
  \brief Constructs a iteration with a predicate.
  \param p Predicate for the iteration.
  */
  iteration_t(Transformer && t, Predicate && p) noexcept :
    transform_{t},
    predicate_{p}
  {}

  /**
  \brief Invokes the predicate of the iteration over a data item.
  */
  template <typename Item>
  bool predicate(Item && item) const {
    return predicate_(std::forward<Item>(item));
  }

  /**
  \brief Applies the transformation over a data item.
  */
  template <typename Item>
  auto transform(Item && item) const {
    return transform_(std::forward<Item>(item));
  }

  /**
  \brief Applies the transformation over a data item.
  */
  template<typename T>
  auto operator()(T && item){
    return transform(item);
  }

private:
  Transformer transform_;
  Predicate predicate_;
};

namespace internal {

template<typename T>
struct is_iteration : std::false_type {};

template<typename T, typename P>
struct is_iteration<iteration_t<T,P>> : std::true_type {};

} // namespace internal

template <typename T>
static constexpr bool is_iteration = internal::is_iteration<std::decay_t<T>>();

template <typename T>
using requires_iteration = typename std::enable_if_t<is_iteration<T>, int>;

}

#endif
