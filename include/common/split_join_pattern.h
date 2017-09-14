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
#ifndef GRPPI_COMMON_SPLIT_JOIN_PATTERN_H
#define GRPPI_COMMON_SPLIT_JOIN_PATTERN_H

#include <type_traits>

namespace grppi {

/**
\brief Representation of farm pattern.
Represents a farm of n replicas from a transformer.
\tparam Transformer Callable type for the farm transformer.
*/
template <typename SplitPolicy, typename ... Transformers>
class split_join_t {
public:

  using transformer_type = std::tuple<Transformers...>;

  /**
  \brief Constructs a farm with a cardinality and a transformer.
  \param n Number of replicas for the farm.
  \param t Transformer for the farm.
  */
  split_join_t(SplitPolicy policy, Transformers && ... t) noexcept :
    policy_{policy}, transformers_{t...}
  {}


  auto get_policy() {
    return policy_;
  }

  template <std::size_t I>
  auto flow() const noexcept {
    return std::get<I>(transformers_);
  }

  /**
  \brief Farm's cardinality or number of replicas.
  \return The farm's cardinality. 
  */
  int num_transformers() const noexcept {
    return sizeof...(Transformers);
  }

private:
  std::tuple<Transformers...> transformers_;
  SplitPolicy policy_;
};

namespace internal {

template <typename, template <typename, typename...> class >
struct is_split_join : std::false_type {};

template <class...F, template <class, class...> class W  >
struct is_split_join < W<F...>, W> : std::true_type {};

} // namespace internal

template <class T>
static constexpr bool is_split_join = internal::is_split_join<T, split_join_t >();

template <class T>
using requires_split_join = typename std::enable_if_t<is_split_join<T>, int>;

}

#endif
