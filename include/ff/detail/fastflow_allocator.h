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
 \brief Fastflow allocator wrapper.
 A fastflow_allocator offers simplified interface for allocating and deallocating
 objects using ff::ff_malloc and ff::ff_free for objects of type T.
 \tparam T type of objects to allocate and deallocate.
 */
template <typename T>
struct fastflow_allocator {

  /**
  \brief Allocate an object and intialize it.
  \param value Value used to initialize the object.
  */
  static T * allocate(const T & value) {
    void * p_buf = ::ff::ff_malloc(sizeof(T));
    T * p_val = new (p_buf) T{value};
  }

  /**
  \brief Deallocate an object that was allocated with allocate.
  \param p_val Pointer that was obtained with a call to allocate()
  */
  static void deallocate(T * p_val) {
    p_val->~T();
    ::ff::ff_free(p_val);
  }
  
};


} // namespace detail_ff

} // namespace grppi

#endif
