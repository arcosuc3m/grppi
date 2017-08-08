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

#ifndef GRPPI_COMMON_ITERATOR_H
#define GRPPI_COMMON_ITERATOR_H

#include <utility>
#include <tuple>

namespace grppi{

namespace internal {

template <typename F, typename T, std::size_t ... I>
decltype(auto) apply_iterator_increment_impl(F && f, T & t, std::index_sequence<I...>)
{
  return std::forward<F>(f)(*std::get<I>(t)++...);
}

} // namespace internal

/**
\brief Applies a callable object to the values obtained from the interators in a tuple.
This function takes callable object `f` and a tuple-like with iterators (e.g.
the result of `make_tuple(it1, it2, it3)`)

and performs the action

~~~{.cpp}
f(*it1++, *it2++, *it3++);
~~~

\tparam Type of the callable object.
\tparam T Tuple type containing a tuple of iterators
\param f Callable object to be invoked.
\param t Tuple of iterators.
\post All iterators in t have been incremented
\post `f` has been invoked with the contents of the iterator in the tuple.
*/
template <typename F, typename T>
decltype(auto) apply_iterators_increment(F && f, T & t)
{
  using tuple_raw_type = std::decay_t<T>;
  constexpr std::size_t size = std::tuple_size<tuple_raw_type>::value;
  return internal::apply_iterator_increment_impl(std::forward<F>(f), t,
      std::make_index_sequence<size>());
}

namespace internal {

template <typename F, typename T, std::size_t ... I>
decltype(auto) apply_iterators_indexed_impl(F && f, T && t, std::size_t i,
    std::index_sequence<I...>)
{
  return std::forward<F>(f)(std::get<I>(t)[i]...);
}

} // namespace internal

/**
\brief Applies a callable object to the values obtained from the iterators in a tuple
by indexing.
This function takes callable object `f`, a tuple-like with iterators (e.g.
the result of `make_tuple(it1, it2, it3)`) and an integral index `i`.

and performs the action

~~~{.cpp}
f(it1[i], it2[i], it3[i]);
~~~

\tparam F Type of the callable object.
\tparam T Tuple type containing a tuple of iterators
\param f Callable object to be invoked.
\param t Tuple of iterators.
\param i Integral index to apply to each interator.
\post All iterators in t have been incremented
\post `f` has been invoked with the contents of the iterator in the tuple.
*/
template <typename F, typename T>
decltype(auto) apply_iterators_indexed(F && f, T && t, std::size_t i)
{
  using tuple_raw_type = std::decay_t<T>;
  constexpr std::size_t size = std::tuple_size<tuple_raw_type>::value;
  return internal::apply_iterators_indexed_impl(std::forward<F>(f), 
      std::forward<T>(t), i, std::make_index_sequence<size>());
}

namespace internal {

template <typename T, std::size_t ... I>
auto iterators_next_impl(T && t, int n, std::index_sequence<I...>) {
  return make_tuple(
    std::next(std::get<I>(t), n)...
  );
}

} // namespace internal

/**
\brief Computes next n steps from a tuple of iterators.
\tparam T Tuple type cotaining a tuple of iterators.
\param t Tuple of iterators.
\param n Number of steps to advance.
\note This function is the equivalent to std::next for a tuple of iterators.
\returns A new tuple with the result iterators.
*/
template <typename T>
auto iterators_next(T && t, int n) {
  using tuple_raw_type = std::decay_t<T>;
  constexpr std::size_t size = std::tuple_size<tuple_raw_type>::value;
  return internal::iterators_next_impl(std::forward<T>(t), n,
      std::make_index_sequence<size>());
}




/// Advance a pack of iterators by a delta.
/// Every iterator in the parameter pack in is increased n steps.
template <typename ... InputIt>
void advance_iterators(size_t delta, InputIt & ... in) {
  // This hack can be done in C++14.
  // It can be simplified in C++17 with folding expressions.
  using type = int[];
  type { 0, (in += delta, 0) ...};
} 

/// Advance a pack of iterators by one unit.
/// Every iterator in the parameter pack in is increased 1 step.
template <typename ... InputIt>
void advance_iterators(InputIt & ... in) {
  // This hack can be done in C++14.
  // It can be simplified in C++17 with folding expressions.
  using type = int[];
  type { 0, (in++, 0) ...};
} 


} // namespace grppi

#endif
