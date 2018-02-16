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
  Transformer & transformer(){
    return transformer_;
  }
  /**
  \brief Invokes the trasnformer of the farm over a data item.
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

template <typename T, typename U>
using concept_farm = typename std::enable_if_t<is_farm<T>, U>;

}

#endif
