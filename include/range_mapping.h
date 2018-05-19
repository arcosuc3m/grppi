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

#include <experimental/type_traits>

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

namespace meta {

template <typename T, template <typename> class E>
using has_member = std::experimental::is_detected<E,T>;

template <typename T, typename R, template <typename> class E>
using has_member_returning = std::experimental::is_detected_convertible<R,E,T>;

template<typename ... Ts> 
struct conjunction : std::true_type {};

template<typename T> 
struct conjunction<T> : T {};

template<class T1, class... Ts>
struct conjunction<T1, Ts...> : std::conditional_t<bool(T1::value), conjunction<Ts...>, T1> {};

template <template <typename> class C, typename ... Ts>
using requires = std::enable_if_t<meta::conjunction<C<Ts>...>::value,int>;

}

template <typename T>
struct range_concept
{
  template <typename U>
  using begin = decltype(std::declval<U>().begin());

  template <typename U>
  using has_begin = meta::has_member<U,begin>;

  template <typename U>
  using end = decltype(std::declval<U>().end());

  template <typename U>
  using has_end = meta::has_member<U,end>;

  template <typename U>
  using size = decltype(std::declval<const U>().size());

  template <typename U>
  using has_size = meta::has_member_returning<U,std::size_t,size>;

  using requires = meta::conjunction<
      has_begin<T>, 
      has_end<T>,
      has_size<T>>;

  static constexpr bool value = requires::value;
};

template <typename T>
using requires_range = meta::requires<range_concept,T>;

template <typename ... Ts>
using requires_ranges = meta::requires<range_concept,Ts...>;

template <typename C,
        requires_range<C> = 0>
range_mapping<C> make_range(C & c) { return {c}; }

template <typename ... Cs,
        requires_ranges<Cs...> = 0>
std::tuple<range_mapping<Cs>...> make_ranges(Cs & ... c) 
{
  return std::make_tuple(range_mapping<Cs>{c}...); 
}

template <typename R>
constexpr bool is_range() { return range_concept<R>::requires::value; }

template <typename ... Rs>
constexpr bool are_ranges() { return meta::conjunction<range_concept<Rs>...>::value; } 

template <typename ... Rs, std::size_t ... I>
std::tuple<typename Rs::iterator...> range_begin_impl(std::tuple<Rs...> rt, std::index_sequence<I...>)
{
  return std::make_tuple(std::get<I>(rt).begin()...);
}

template <typename ... Rs,
        requires_ranges<Rs...> = 0>
std::tuple<typename Rs::iterator...> range_begin(std::tuple<Rs...> rt) 
{
  return range_begin_impl(rt, std::make_index_sequence<sizeof...(Rs)>{});
}

template <typename ... Rs,
        requires_ranges<Rs...> = 0>
auto range_size(std::tuple<Rs...> rt)
{
  return std::get<0>(rt).size();
}


}

#endif
