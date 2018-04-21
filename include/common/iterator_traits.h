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
#ifndef GRPPI_COMMON_ITERATOR_TRAITS_H
#define GRPPI_COMMON_ITERATOR_TRAITS_H

namespace grppi{

namespace internal {

template<typename T, typename = void>
struct is_iterator
{
  static constexpr bool value = false;
};

template<typename T>
struct is_iterator<T, typename std::enable_if<!std::is_same<typename std::iterator_traits<T>::value_type, void>::value>::type>
{
  static constexpr bool value = true;
};

template<typename T, typename ...other_T>
struct are_iterators
{
  static constexpr bool value = is_iterator<T>::value && are_iterators<other_T...>::value;
};

template<typename T>
struct are_iterators<T>
{
  static constexpr bool value = is_iterator<T>::value;
};

}

template <typename T>
constexpr bool is_iterator = internal::is_iterator<T>::value;

template <typename T>
using requires_iterator = std::enable_if_t<is_iterator<T>, int>;

template<typename ...T>
constexpr bool are_iterators = internal::are_iterators<T...>::value;

template<typename ...T>
using requires_iterators = std::enable_if_t<are_iterators<T...>, int>;

}

#endif
