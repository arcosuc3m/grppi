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
#ifndef GRPPI_COMMON_META_H
#define GRPPI_COMMON_META_H

#include <type_traits>

#if __has_include(<experimental/type_traits>)

#  include <experimental/type_traits>

#  ifndef __cpp_lib_experimental_detect
#    error "C++ detection idiom not supported. Upgrade your C++ compiler"
#  endif
#else
#  error "Experimental type traits not found. Upgrade your C++ compiler."
#endif

namespace grppi::meta {

/**
\brief Detects if template E<Ts...> is valid.
\tparam E Expression type.
\tparam Ts Types used as arguments for E.
*/
  template<template<typename...> class E, typename ... Ts>
  using is_detected = std::experimental::is_detected<E, Ts...>;

/**
\brief Detects if template E<Ts...> is valid and convertible to R.
\tparam R Type to which the expression is convertible.
\tparam E Expression type.
\tparam Ts Types used as arguments for E.
*/
  template<typename R, template<typename...> class E, typename ... Ts>
  using is_detected_convertible = std::experimental::is_detected_convertible<R, E, Ts...>;

/**
\brief Checks if a pack of types all satisfy concept C.
\tparam C Concept to be checked.
\tparam Ts Types to be checked against concept.
If the concept is satisfied has a public typedef type (equal to int), otherwise 
there is no type defined.
*/
  template<template<typename> class C, typename ... Ts>
  using requires_ = std::enable_if_t<std::conjunction<C<Ts>...>::value, int>;

  /**
   * \brief Get the output type for N-th invocable object in a tuple.
   * \tparam N Index in the tuple for the invocable.
   * \tparam T Tuple of invocable objects.
   * \tparam A Argument type for the first invocable object.
   */
  template<std::size_t N, typename T, typename A>
  struct get_output;

  /**
   * \brief Alias for getting output type of N-th invocable object in tuple.
   * \note Equivalent to get_output<N,T,A>::type.
   * \tparam N Index in the tuple for the invocable.
   * \tparam T Tuple of invocable objects.
   * \tparam A Argument type for the first invocable object.
   */
  template<std::size_t N, typename T, typename A>
  using get_output_t = typename get_output<N, T, A>::type;

  template<typename T, typename ... Us, typename A>
  struct get_output<0, std::tuple<T, Us...>, A> {
    using operation_type = std::tuple_element_t<0, std::tuple<T, Us...>>;
    using type = std::invoke_result_t<operation_type, A>;
  };

  template<std::size_t N, typename T, typename ... Us, typename A>
  struct get_output<N, std::tuple<T, Us...>, A> {
    using operation_type = std::tuple_element_t<N, std::tuple<T, Us...>>;
    using Arg = get_output_t<N - 1, std::tuple<T, Us...>, A>;
    using type = std::invoke_result_t<operation_type, Arg>;
  };

  /**
   * \brief Get the input type for N-th invocable object in a tuple.
   * \tparam N Index in the tuple for the invocable.
   * \tparam T Tuple of invocable objects.
   * \tparam A Argument type for the first invocable object.
   */
  template<std::size_t N, typename T, typename A>
  struct get_input;

  /**
   * \brief Alias for getting input type of N-th invocable object in tuple.
   * \note Equivalent to get_input<N,T,A>::type.
   * \tparam N Index in the tuple for the invocable.
   * \tparam T Tuple of invocable objects.
   * \tparam A Argument type for the first invocable object.
   */
  template<std::size_t N, typename T, typename A>
  using get_input_t = typename get_input<N, T, A>::type;

  template<typename T, typename ... Us, typename A>
  struct get_input<0, std::tuple<T, Us...>, A> {
    using type = A;
  };

  template<std::size_t N, typename T, typename ... Us, typename A>
  struct get_input<N, std::tuple<T, Us...>, A> {
    using type = get_output<N - 1, std::tuple<T, Us...>, A>;
  };

  template<typename Ret, typename Arg, typename... Rest>
  Arg input_type_t(Ret(*) (Arg, Rest...));

  template<typename Ret, typename F, typename Arg, typename... Rest>
  Arg input_type_t(Ret(F::*) (Arg, Rest...));

  template<typename Ret, typename F, typename Arg, typename... Rest>
  Arg input_type_t(Ret(F::*) (Arg, Rest...) const);

  template <typename F, class = std::enable_if_t<!std::is_pointer<F>::value>>
  decltype(input_type_t(&F::operator())) input_type_l(F);

  template <typename F, class = std::enable_if_t<std::is_pointer<F>::value>>
  decltype(input_type_t(std::declval<F>())) input_type_l(F);

  template <typename F>
  using input_type = decltype(input_type_l(std::declval<F>()));

  template <typename F>
  using output_type = std::invoke_result_t<F,input_type<F>>;


}

#endif
