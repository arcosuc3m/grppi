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

#ifndef GRPPI_COMMON_ITERATOR_H
#define GRPPI_COMMON_ITERATOR_H

namespace grppi{

/// Advance a pack of iterators by a delta.
/// Every iterator in the parameter pack in is increased n steps.
template <typename ... InputIt>
void advance_iterators(size_t delta, InputIt & ... in) {
  // This hack can be done in C++14.
  // It can be simplified in C++17 with folding expressions.
  using type = int[];
  type { 0, (in += delta, 0) ...};
} 

/// Advance a pack of iterators by one unit.
/// Every iterator in the parameter pack in is increased 1 step.
template <typename ... InputIt>
void advance_iterators(InputIt & ... in) {
  // This hack can be done in C++14.
  // It can be simplified in C++17 with folding expressions.
  using type = int[];
  type { 0, (in++, 0) ...};
} 


}

#endif
