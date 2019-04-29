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
#ifndef GRPPI_FF_DETAIL_FASTFLOW_ALLOCATOR_H
#define GRPPI_FF_DETAIL_FASTFLOW_ALLOCATOR_H

#include <ff/allocator.hpp>

namespace grppi {

namespace detail_ff {

/**
 \brief Tag type for fastflow allocation.
 This empty type is used to select the overloaded placement new and delete
 operators that invoke the fastflow allocation routines.
 */
struct ff_arena_t {};

/**
 \brief Fastflow arena object.
 This object will be passed to placement new/delete to use FastFlow allocation strategy.
 */
constexpr ff_arena_t ff_arena;


} // namespace detail_ff

} // namespace grppi

/**
 \brief Placement new for the FastFlow arena.
 Invokes ff_malloc().
 Use by calling new (ff_arena) your_type;
 */
inline void * operator new(
        std::size_t sz, 
        const grppi::detail_ff::ff_arena_t &) noexcept
{
  return ::ff::ff_malloc(sz);
}

/**
 \brief Placement delete for the FastFlow arena.
 Invokes ff_free().
 Use by calling operator delete(ptr,ff_arena);
 */
inline void operator delete(
        void * ptr,
        const grppi::detail_ff::ff_arena_t &) noexcept
{
  ::ff::ff_free(ptr);
}        

#endif
