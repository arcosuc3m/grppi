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
#ifndef GRPPI_COMMON_ITERATION_MULTOUT_PATTERN_H
#define GRPPI_COMMON_ITERATION_MULTOUT_PATTERN_H

#include <type_traits>

namespace grppi {

/**
\brief Representation of iteration pattern.
Represents a iteration that can be used as a stage on a pipeline.
\tparam Predicate Callable type for the iteration predicate.
*/
template <typename Transformer, typename Predicate, typename Guard>
class iteration_multout_t {
public:

  /**
  \brief Constructs a iteration with a predicate.
  \param p Predicate for the iteration.
  */
  iteration_multout_t(Transformer && t, Predicate && p, Guard && g) noexcept :
    transform_{t},
    predicate_{p},
    output_guard_{g}
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

  template <typename Item>
  bool output_guard(Item && item) const {
    return output_guard_(std::forward<Item>(item)); 
  }

private:
  Transformer transform_;
  Predicate predicate_;
  Guard output_guard_;
};

namespace internal {

template<typename T>
struct is_iteration_multout : std::false_type {};

template<typename T, typename P, typename G>
struct is_iteration_multout<iteration_multout_t<T,P,G>> : std::true_type {};

} // namespace internal

template <typename T>
static constexpr bool is_iteration_multout = internal::is_iteration_multout<std::decay_t<T>>();

template <typename T>
using requires_iteration_multout = typename std::enable_if_t<is_iteration_multout<T>, int>;

}

#endif
