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

#include "common/range_concept.h"

namespace grppi {

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

template <typename ... Rs>
class zip_view {
public:
  zip_view(Rs& ... rs) : rngs_{rs...} {}

  auto begin() { return begin_impl(std::make_index_sequence<sizeof...(Rs)>{}); }

  auto size() {
    return std::get<0>(rngs_).size();
  }

private:
  std::tuple<Rs&...> rngs_;

private:

  template <std::size_t ... I>
  auto begin_impl(std::index_sequence<I...>) {
    return std::make_tuple(std::get<I>(rngs_).begin()...);
  }
};

template <typename ... Rs>
auto zip(Rs & ... rs) { return zip_view<Rs...>(rs...); }

}

#endif
