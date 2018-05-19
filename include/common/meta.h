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
#include <experimental/type_traits>
#  ifndef __cpp_lib_experimental_detect
#    error "C++ detection idiom not supported. Upgrade your C++ compiler"
#  endif
#else
#error "Experimental type traits not found. Upgrade your C++ compiler."
#endif

namespace grppi {

namespace meta {

template <template <typename> class E, typename T>
using is_detected = std::experimental::is_detected<E,T>;

template <typename R, template <typename> class E, typename T>
using is_detected_convertible = std::experimental::is_detected_convertible<R,E,T>;

template<typename ... Ts> 
struct conjunction : std::true_type {};

template<typename T> 
struct conjunction<T> : T {};

template<class T1, class... Ts>
struct conjunction<T1, Ts...> : std::conditional_t<bool(T1::value), conjunction<Ts...>, T1> {};

template <template <typename> class C, typename ... Ts>
using requires = std::enable_if_t<meta::conjunction<C<Ts>...>::value,int>;

}

}

#endif
