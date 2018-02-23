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

#ifndef GRPPI_FF_DETAIL_UNORDERED_STREAM_REDUCE_H
#define GRPPI_FF_DETAIL_UNORDERED_STREAM_REDUCE_H

#include "reduce_nodes.h"

#include <ff/farm.hpp>

namespace grppi {

namespace detail_ff {

template <typename Item, typename Reducer, typename Combiner>
class unordered_stream_reduce : public ff::ff_farm<> {
public:

  unordered_stream_reduce(Reducer && reducer, int num_workers);

private:
  std::vector<ff::ff_node*> workers_;

  using emitter_type = reduce_emitter<Item,Reducer>;
  std::unique_ptr<emitter_type> p_emitter_;

  std::unique_ptr<reduce_collector> p_collector_;
};

template <typename Item, typename Reducer, typename Combiner>
unordered_stream_reduce<Item,Reducer,Combiner>::unordered_stream_reduce(
        Reducer && reducer, 
        int num_workers) 
:
 ff_farm<>{false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, 
      static_cast<std::size_t>(num_workers)},
  workers_{},
  p_emitter_{std::make_unique<emitter_type>(reducer.window_size(), reducer.offset())},
  p_collector_{std::make_unique<reduce_collector>()}
{
    for (int i=0; i<num_workers; ++i) {
      workers.push_back(new reduce_worker<Item,Combiner>{
          std::forward<Combiner>(reducer.combiner())});
    }
    this->add_workers(workers_);
    this->add_emitter(p_emitter_.get());
    this->add_collector(p_collector_.get());
  }

} // namespace detail_ff

} // namespace grppi

#endif
