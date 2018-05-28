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
#ifndef GRPPI_COMMON_CALLABLE_TRAITS_H
#define GRPPI_COMMON_CALLABLE_TRAITS_H

#include <type_traits>

namespace grppi {

namespace internal {

/// Identity type trait
template <typename T>
struct identity {
  using type = T;
};

// Callable helper for function objects defers to pointer to call operator
template <typename T>
struct callable_helper : callable_helper<decltype(&T::operator())>
{};

// Callable helper for function type
template <typename R, typename ... Args>
struct callable_helper<R(Args...)> : identity<R(Args...)>
{
  using arity = typename std::integral_constant<size_t, sizeof...(Args)>::type;
};

// Callable helper for pointer to function defers to callable helper for
// function type
template <typename R, typename ... Args>
struct callable_helper<R(*)(Args...)> : callable_helper<R(Args...)>
{};

// Callable helper for pointer to const member function defers to callable helper
// for function type
template <typename C, typename R, typename ... Args>
struct callable_helper<R(C::*)(Args...) const> : callable_helper<R(Args...)>
{};

// Callable helper for pointer to non-const member function defers to callable helper
// for function type
template <typename C, typename R, typename ... Args>
struct callable_helper<R(C::*)(Args...)> : callable_helper<R(Args...)>
{};

// Callable single interface defers to corresponding callable helper
template <typename T>
struct callable : callable_helper<T>
{};

// Convenience meta-function for getting arity of a callable
template <typename T>
constexpr size_t callable_arity() {
  return typename callable<T>::arity();
}

// Meta-function for determining if a callable returns void
template <typename G>
constexpr bool has_void_return() {
  return std::is_same<void,
      typename std::result_of<G()>::type
  >::value;
}

template <typename F, typename I>
constexpr bool has_void_return() {
  return std::is_same<void,
        typename std::result_of<F(I)>::type
      >::value;
}

// Meta-function for determining if a callable has arguments
template <typename F>
constexpr bool has_arguments() {
  return typename internal::callable<F>::arity() != 0;
}

} // end namespace internal


// Meta-function for determining if F is consumer of I
template <typename F, typename I>
constexpr bool is_consumer = internal::has_void_return<F,I>();

// Meta-function for determining if G is a generator
template <typename G>
constexpr bool is_generator = 
  !internal::has_void_return<G>();

// Concept emulation requiring a callable generating values
template <typename G>
using requires_generator =
  typename std::enable_if_t<is_generator<G>, int>;


// Concept emulation requiring a callable with one or more arguments
template <typename F>
using requires_arguments =
  typename std::enable_if_t<internal::has_arguments<F>(), int>;

// Concept emulation requiring a callable consuming values
template <typename F, typename I>
using requires_consumer =
  typename std::enable_if_t<internal::has_void_return<F(I)>(), int>;



} // end namespace grppi

#endif
