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
#ifndef GRPPI_COMMON_FARM_PATTERN_H
#define GRPPI_COMMON_FARM_PATTERN_H

#include <type_traits>

namespace grppi {

/**
\brief Representation of farm pattern.
Represents a farm of n replicas from a transformer.
\tparam Transformer Callable type for the farm transformer.
*/
template <typename Transformer>
class farm_t {
public:

  using transformer_type = Transformer;

  /**
  \brief Constructs a farm with a cardinality and a transformer.
  \param n Number of replicas for the farm.
  \param t Transformer for the farm.
  */
  farm_t(int n, Transformer && t) noexcept :
    cardinality_{n}, transformer_{t}
  {}

  /**
  \brief Farm's cardinality or number of replicas.
  \return The farm's cardinality. 
  */
  int cardinality() const noexcept {
    return cardinality_;
  }

  /**
  \brief Return the transformer function stored in the farm pattern.
  \return The transformer function. 
  */
  Transformer & transformer() & noexcept {
    return transformer_;
  }

  auto && transformer() && noexcept {
    return transformer_;
  }

  /**
  \brief Invokes the transformer of the farm over a data item.
  */
  template <typename I>
  auto operator()(I && item) const {
    return transformer_(std::forward<I>(item));
  }

private:
  int cardinality_;
  Transformer transformer_;
};

namespace internal {

template<typename T>
struct is_farm : std::false_type {};

template<typename T>
struct is_farm<farm_t<T>> : std::true_type {};

} // namespace internal

template <typename T>
static constexpr bool is_farm = internal::is_farm<std::decay_t<T>>();

template <typename T>
using requires_farm = typename std::enable_if_t<is_farm<T>, int>;

}

#endif
