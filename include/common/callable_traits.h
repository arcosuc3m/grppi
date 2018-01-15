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

#ifndef GRPPI_COMMON_CALLABLE_TRAITS_H
#define GRPPI_COMMON_CALLABLE_TRAITS_H

#include <type_traits>

namespace grppi {

namespace internal {

//Function trait to determine argument types
template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
{
    enum { arity = sizeof...(Args) };

    enum { callable = true };     
    typedef ReturnType result_type;

    template <size_t i>
    struct arg
    {
        typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
    };
};

template<typename T>
struct is_callable_impl {
private:
    typedef char(&yes)[1];
    typedef char(&no)[2];

    struct Fallback { void operator()(); };
    struct Derived : T, Fallback { };

    template<typename U, U> struct Check;

    template<typename>
    static yes test(...);

    template<typename C>
    static no test(Check<void (Fallback::*)(), &C::operator()>*);

public:
    static const bool value = sizeof(test<Derived>(0)) == sizeof(yes);
};


template<typename T>
struct is_callable_type
    : std::conditional<
        std::is_class<T>::value,
        is_callable_impl<T>,
        std::false_type
    >::type
{ };


template<typename T, typename = void>
struct is_callable : std::is_function<T> { };

template <typename T>
struct is_callable<T, typename std::enable_if<
    std::is_same<decltype(void(&T::operator())), void>::value
    >::type> : std::true_type { };



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
