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

#if __has_include(<experimental/type_traits>)
#  include <experimental/type_traits>
#  ifdef _LIBCPP_VERSION
#    if _LIBCPP_VERSION < 5000
#      error "C++ detection idiom not supported. Upgrade your C++ standard library"
#    endif
#  else
#    if __cpp_lib_experimental_detect < 201505
#      error "C++ detection idiom not supported. Upgrade your C++ compiler/standard library"
#    endif
#  endif
#else
#  error "Experimental type traits not found. Upgrade your C++ compiler."
#endif

namespace grppi {

namespace meta {

/**
\brief Detects if template E<Ts...> is valid.
\tparam E Expression type.
\tparam Ts Types used as arguments for E.
*/
template <template <typename...> class E, typename ... Ts>
using is_detected = std::experimental::is_detected<E,Ts...>;

/**
\brief Detects if template E<Ts...> is valid and convertible to R.
\tparam R Type to which the expression is convertible.
\tparam E Expression type.
\tparam Ts Types used as arguments for E.
*/
template <typename R, template <typename...> class E, typename ... Ts>
using is_detected_convertible = std::experimental::is_detected_convertible<R,E,Ts...>;


/**
\brief Forms the logical conjunction of several traits.
\tparam Ts Type traits to be conjuncted.
*/
template<typename ... Ts> 
struct conjunction : std::true_type {};

/**
\brief Specialization of conjunction for a single trait.
\tparam T Type trait to be conjuncted.
*/
template<typename T> 
struct conjunction<T> : T {};

/**
\brief Specialization of conjunction for multiple traits.
\tparam T1 First type trait.
\tparam Ts Rest of type traits.
*/
template<class T1, class... Ts>
struct conjunction<T1, Ts...> : std::conditional_t<bool(T1::value), conjunction<Ts...>, T1> {};

/**
\brief Checks if a pack of types all satisfy concept C.
\tparam C Concept to be checked.
\tparam Ts Types to be checked against concept.
If the concept is satisfied has a public typedef type (equal to int), otherwise 
there is no type defined.
*/
template <template <typename> class C, typename ... Ts>
using requires = std::enable_if_t<conjunction<C<Ts>...>::value,int>;

}

}

#endif
