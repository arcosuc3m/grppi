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

template <typename C>
range_mapping<C> make_range(C & c) { return {c}; }

template <typename ... Cs>
std::tuple<range_mapping<Cs>...> make_ranges(Cs & ... c) 
{ 
  return std::make_tuple(range_mapping<Cs>{c}...); 
}


template <typename R>
struct is_range_impl
{
  static constexpr bool value = false;
};

template <typename C>
struct is_range_impl<range_mapping<C>> {
  static constexpr bool value = true;
};

template <typename R, typename ... Rs>
struct are_ranges_impl
{
  static constexpr bool value = is_range_impl<R>::value && are_ranges_impl<Rs...>::value;
};

template <typename R>
struct are_ranges_impl<R>
{
  static constexpr bool value = is_range_impl<R>::value;
};


template <typename R>
constexpr bool is_range() { return is_range_impl<R>::value; }

template <typename ... Rs>
constexpr bool are_ranges() { return are_ranges_impl<Rs...>::value; } 

template <typename R>
using requires_range = std::enable_if_t<is_range<R>(), int>;

template <typename ... Rs>
using requires_ranges = std::enable_if_t<are_ranges<Rs...>(), int>;

template <typename ... Rs, std::size_t ... I>
std::tuple<typename Rs::iterator...> range_begin_impl(std::tuple<Rs...> rt, std::index_sequence<I...>)
{
  return std::make_tuple(std::get<I>(rt).begin()...);
}

template <typename ... Rs>
std::tuple<typename Rs::iterator...> range_begin(std::tuple<Rs...> rt) 
{
  return range_begin_impl(rt, std::make_index_sequence<sizeof...(Rs)>{});
}

template <typename ... Rs>
auto range_size(std::tuple<Rs...> rt)
{
  return std::get<0>(rt).size();
}


}

#endif
