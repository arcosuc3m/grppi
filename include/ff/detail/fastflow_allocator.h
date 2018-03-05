/**
* @version		GrPPI v0.3
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
