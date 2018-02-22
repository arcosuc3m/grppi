/**
* @version    GrPPI v0.3
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
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

#ifndef GRPPI_FF_DETAIL_ITERATION_WORKER_H
#define GRPPI_FF_DETAIL_ITERATION_WORKER_H

#ifdef GRPPI_FF

#include <vector>

#include <ff/node.hpp>

namespace grppi {

namespace detail_ff {

template<typename Item, typename Iteration>
class iteration_worker : public ff::ff_node_t<Item> {
public:
  iteration_worker(Iteration && iteration) :
      iteration_{std::move(iteration)}
  {}

  Item * svc(Item * p_item) {
    do {
      *p_item = iteration_.transform(*p_item);
    }
    while (!iteration_.predicate(*p_item));
    return p_item;
  }

private:
  Iteration iteration_;
};

} // namespace detail_ff

} // namespace grppi

#else

#endif // GRPPI_FF

#endif
