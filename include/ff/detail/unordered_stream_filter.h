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

#ifndef GRPPI_FF_DETAIL_UNORDERED_STREAM_FILTER_H
#define GRPPI_FF_DETAIL_UNORDERED_STREAM_FILTER_H

#ifdef GRPPI_FF

#include "nodes.h"

#include <ff/farm.hpp>
#include <ff/allocator.hpp>

namespace grppi {

namespace detail_ff {

template <typename Item, typename Filter>
class unordered_stream_filter : public ff::ff_ofarm {
public:
  unordered_stream_filter(Filter && filter, int num_workers) :
      ff::ff_ofarm{false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, num_workers},
      filter_{filter},
      concurrency_degree_{num_workers},
      workers_{},
      p_collector_{nullptr}
  {
    for(int i=0;i<num_workers;++i)
      workers_.push_back(new unordered_filter_worker<Item,Filter>{
        std::forward<Filter>(filter_)});
    add_workers(workers_);
    p_emitter_ = new unordered_filter_emitter<Item>{};
    add_emitter(p_emitter_);
    p_collector_ = new unordered_filter_collector<Item>{};
    add_collector(p_collector_);
  }

  ~unordered_stream_filter() {
    delete p_emitter_;
    delete p_collector_;
  }

private:

  Filter filter_;
  int concurrency_degree_;
  std::vector<ff::ff_node *> workers_;
  unordered_filter_emitter<Item> * p_emitter_;
  unordered_filter_collector<Item> * p_collector_;
};


} // namespace detail_ff

} // namespace grppi

#else

#endif // GRPPI_FF

#endif
