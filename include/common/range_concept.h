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
#ifndef GRPPI_COMMON_RANGE_CONCEPT_H
#define GRPPI_COMMON_RANGE_CONCEPT_H

#include "common/meta.h"

namespace grppi {

/** 
\addtogroup concepts
@{
*/

/**
\brief Concept for ranges.
A range is any type that:
  - Has a begin() member.
  - Has an end() member.
  - Has a size() member returning a value convertible to std::size_t.
*/
template <typename T>
struct range_concept
{
  /// Return type of U::begin().
  template <typename U>
  using begin = decltype(std::declval<U>().begin());

  /// Detects if a type has begin() member function.
  template <typename U>
  using has_begin = meta::is_detected<begin,U>;

  /// Return type of U::end().
  template <typename U>
  using end = decltype(std::declval<U>().end());

  /// Detects if a type has end() member function.
  template <typename U>
  using has_end = meta::is_detected<end,U>;

  /// Return type of U::size().
  template <typename U>
  using size = decltype(std::declval<const U>().size());

  /// Detects if a type has a size() member function returning std::size_t.
  template <typename U>
  using has_size = meta::is_detected_convertible<std::size_t,size,U>;

  /// Requirements for range_concept.
  using requires = meta::conjunction<
      has_begin<T>, 
      has_end<T>,
      has_size<T>>;

  /// Boolean value for the range_concept requirements.
  static constexpr bool value = requires::value;
};

/**
@}
*/

}

#endif
