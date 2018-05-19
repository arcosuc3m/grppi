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
#ifndef GRPPI_RANGE_MAPPING_H
#define GRPPI_RANGE_MAPPING_H

#include "common/meta.h"

namespace grppi {

template <typename T>
struct range_concept
{
  template <typename U>
  using begin = decltype(std::declval<U>().begin());

  template <typename U>
  using has_begin = meta::is_detected<begin,U>;

  template <typename U>
  using end = decltype(std::declval<U>().end());

  template <typename U>
  using has_end = meta::is_detected<end,U>;

  template <typename U>
  using size = decltype(std::declval<const U>().size());

  template <typename U>
  using has_size = meta::is_detected_convertible<std::size_t,size,U>;

  using requires = meta::conjunction<
      has_begin<T>, 
      has_end<T>,
      has_size<T>>;

  static constexpr bool value = requires::value;
};

template <typename ... Rs, std::size_t ... I>
std::tuple<typename Rs::iterator...> range_begin_impl(std::tuple<Rs...> rt, std::index_sequence<I...>)
{
  return std::make_tuple(std::get<I>(rt).begin()...);
}

template <typename ... Rs,
        meta::requires<range_concept,Rs...> = 0>
std::tuple<typename Rs::iterator...> range_begin(std::tuple<Rs...> rt) 
{
  return range_begin_impl(rt, std::make_index_sequence<sizeof...(Rs)>{});
}

template <typename ... Rs,
        meta::requires<range_concept,Rs...> = 0>
auto range_size(std::tuple<Rs...> rt)
{
  return std::get<0>(rt).size();
}

template <typename C>
class range_mapping {
public:
  using iterator = typename C::iterator;

  range_mapping(C & c) : first_{c.begin()}, last_{c.end()} {}
  range_mapping(iterator f, iterator l) : first_{f}, last_{l} {}

  auto begin() { return first_; }
  auto end() { return last_; }
  auto size() const { return std::distance(first_,last_); }

private:
  iterator first_, last_;
};

template <typename C,
        meta::requires<range_concept,C> = 0>
range_mapping<C> make_range(C & c) { return {c}; }

template <typename ... Cs,
        meta::requires<range_concept,Cs...> = 0>
std::tuple<range_mapping<Cs>...> make_ranges(Cs & ... c) 
{
  return std::make_tuple(range_mapping<Cs>{c}...); 
}

}

#endif
