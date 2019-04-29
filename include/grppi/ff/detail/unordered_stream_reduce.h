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
