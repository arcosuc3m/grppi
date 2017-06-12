/**
* @version		GrPPI v0.2
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

#ifndef GRPPI_CALLABLE_TRAITS_H
#define GRPPI_CALLABLE_TRAITS_H

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

// Callalble helper for pointer to const member function defers to callable helper
// for function type
template <typename C, typename R, typename ... Args>
struct callable_helper<R(C::*)(Args...) const> : callable_helper<R(Args...)>
{};

// Callalble helper for pointer to non-const member function defers to callable helper
// for function type
template <typename C, typename R, typename ... Args>
struct callable_helper<R(C::*)(Args...)> : callable_helper<R(Args...)>
{};

// Callable single interface defers to correspondign callable helper
template <typename T>
struct callable : callable_helper<T>
{};

// Convenience meta-function for getting arity of a callable
template <typename T>
constexpr size_t callable_arity() {
  return typename callable<T>::arity();
}

// Meta-function for determining if a callable has arguments
template <typename F>
constexpr bool has_arguments() {
  return typename callable<F>::arity() != 0;
}

} // end namespace internal

// Concept emulation requiring a callable with no arguments
template <typename F>
using requires_no_arguments =
  typename std::enable_if_t<!internal::has_arguments<F>(), int>;

// Concept emulation requiring a callable with one or more arguments
template <typename F>
using requires_arguments =
  typename std::enable_if_t<internal::has_arguments<F>(), int>;




} // end namespace grppi

#endif
