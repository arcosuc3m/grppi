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
#ifndef GRPPI_COMMON_EXECUTION_TRAITS_H
#define GRPPI_COMMON_EXECUTION_TRAITS_H

#include <type_traits>

namespace grppi {

/**
\brief Determines if an execution policy is supported in the current compilation.
\note This must be specialized by every execution policy.
*/
template <typename E>
constexpr bool is_supported() { return false; }

template <typename E>
using requires_execution_supported =
  std::enable_if_t<is_supported<E>(), int>;

template <typename E>
using requires_execution_not_supported =
  std::enable_if_t<!is_supported<E>(), int>;

/**
\brief Determines if an execution policy supports the map pattern.
\note This must be specialized by every execution policy supporting the pattern.
*/
template <typename E>
constexpr bool supports_map() { return false; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note This must be specialized by every execution policy supporting the pattern.
*/
template <typename E>
constexpr bool supports_reduce() { return false; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note This must be specialized by every execution policy supporting the pattern.
*/
template <typename E>
constexpr bool supports_map_reduce() { return false; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note This must be specialized by every execution policy supporting the pattern.
*/
template <typename E>
constexpr bool supports_stencil() { return false; }

/**
\brief Determines if an execution policy supports the divide-conquer pattern.
\note This must be specialized by every execution policy supporting the pattern.
*/
template <typename E>
constexpr bool supports_divide_conquer() { return false; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note This must be specialized by every execution policy supporting the pattern.
*/
template <typename E>
constexpr bool supports_pipeline() { return false; }

/**
\brief Determines if an execution policy supports the split-join pattern.
\note This must be specialized by every execution policy supporting the pattern.
*/
template <typename E>
constexpr bool supports_stream_pool() { return false; }

} // end namespace grppi
#endif
