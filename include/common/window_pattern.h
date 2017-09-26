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
#ifndef GRPPI_COMMON_WINDOW_PATTERN_H
#define GRPPI_COMMON_WINDOW_PATTERN_H

#include <type_traits>
#include <memory>

namespace grppi {

/**
\brief Representation of farm pattern.
Represents a farm of n replicas from a transformer.
\tparam Transformer Callable type for the farm transformer.
*/
template <typename Window>
class window_t {
public:

  using window_type = Window;
  using input_type = typename window_type::item_type;
  /**
  \brief Constructs a farm with a cardinality and a transformer.
  \param n Number of replicas for the farm.
  \param t Transformer for the farm.
  */
  window_t(Window && win) noexcept :
    window_{win}
  {}

/*  window_t(Window &win) noexcept :
    window_{win}
  {}*/

  auto& get_window(){
    return window_;
  }

  void save_window(window_type w)
  {
    window_ = w;
  }

private:
  Window window_;
};


namespace internal {


template <typename, template <typename, typename...> class >
struct is_window : std::false_type {};


template <class...F, template <class, class...> class W  >
struct is_window < W<F...>, W> : std::true_type {};

} // namespace internal

template <class F>
static constexpr bool is_window = internal::is_window< std::decay_t<F>, window_t>();

template <class F>
using requires_window = typename std::enable_if_t<is_window<F>, int>;


}

#endif
