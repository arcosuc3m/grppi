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
#ifndef GRPPI_COMMON_FILTER_PATTERN_H
#define GRPPI_COMMON_FILTER_PATTERN_H

#include <type_traits>
#include "meta.h"

namespace grppi {

/**
\brief Representation of filter pattern.
Represents a filter that can be used as a stage on a pipeline.
\tparam Predicate Callable type for the filter predicate.
*/
template <typename Predicate>
class filter_t {
public:

  using predicate_type = Predicate;
  using input_type = meta::input_type<Predicate>;

  /**
  \brief Constructs a filter with a predicate.
  \param p Predicate for the filter.
  */
  filter_t(Predicate && p, bool keep) noexcept :
    predicate_{p},
    keep_{keep}
  {}

  /**
  \brief Invokes the predicate of the filter over a data item.
  */
  template <typename I>
  bool operator()(I && item) const {
    return predicate_(std::forward<I>(item));
  }

  bool keep() const { return keep_; }

private:
  Predicate predicate_;
  bool keep_;
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

}

#endif
