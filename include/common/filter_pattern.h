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
#ifndef GRPPI_COMMON_FILTER_PATTERN_H
#define GRPPI_COMMON_FILTER_PATTERN_H

#include <type_traits>

namespace grppi {

/**
\brief Representation of filter pattern.
Represents a filter that can be used as a stage on a pipeline.
\tparam Predicate Callable type for the filter predicate.
*/
template <typename Predicate>
class filter_t {
public:

  /**
  \brief Constructs a filter with a predicate.
  \param p Predicate for the filter.
  */
  filter_t(Predicate && p) noexcept :
    predicate_{p}
  {}

  /**
  \brief Invokes the predicate of the filter over a data item.
  */
  template <typename I>
  bool operator()(I && item) const {
    return predicate_(std::forward<I>(item));
  }

private:
  Predicate predicate_;
};

namespace internal {

template<typename T>
struct is_filter : std::false_type {};

template<typename T>
struct is_filter<filter_t<T>> : std::true_type {};

} // namespace internal

template <typename T>
static constexpr bool is_filter = internal::is_filter<std::decay_t<T>>();

template <typename T>
using requires_filter = typename std::enable_if_t<is_filter<T>, int>;

template <typename T, typename U>
using concept_filter = typename std::enable_if_t<is_filter<T>, U>;

}

#endif
