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
#ifndef GRPPI_COMMON_WINDOWED_FARM_PATTERN_H
#define GRPPI_COMMON_WINDOWED_FARM_PATTERN_H

#include <type_traits>
#include <memory>

namespace grppi {

/**
\brief Representation of farm pattern.
Represents a farm of n replicas from a transformer.
\tparam Transformer Callable type for the farm transformer.
*/
template <typename Transformer,typename Window>
class window_farm_t {
public:

  using transformer_type = Transformer;
  using window_type = Window;

  /**
  \brief Constructs a farm with a cardinality and a transformer.
  \param n Number of replicas for the farm.
  \param t Transformer for the farm.
  */
  window_farm_t(int n, Transformer && t, Window && win) noexcept :
    cardinality_{n}, transformer_{t}, window_{win}
  {}

  /**
  \brief Farm's cardinality or number of replicas.
  \return The farm's cardinality. 
  */
  int cardinality() const noexcept {
    return cardinality_;
  }
  
  Window& get_window(){
    return window_;
  }

  void save_window(window_type w)
  {
    window_ = w;
  }

  /**
  \brief Invokes the trasnformer of the farm over a data item.
  */
  template <typename I>
  auto transform(I && item) const {
    return transformer_(std::forward<I>(item));
  }

private:
  int cardinality_;
  Transformer transformer_;
  Window window_;
};


namespace internal {


template <typename, template <typename, typename...> class >
struct is_window_farm : std::false_type {};


template <class...F, template <class, class...> class W  >
struct is_window_farm < W<F...>, W> : std::true_type {};

} // namespace internal

template <class F>
static constexpr bool is_window_farm = internal::is_window_farm< F, window_farm_t>();

template <class F>
using requires_window_farm = typename std::enable_if_t<is_window_farm<F>, int>;


}

#endif
