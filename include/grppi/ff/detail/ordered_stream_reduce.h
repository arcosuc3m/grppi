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
#ifndef GRPPI_FF_DETAIL_ORDERED_STREAM_REDUCE_H
#define GRPPI_FF_DETAIL_ORDERED_STREAM_REDUCE_H

#include "reduce_nodes.h"

#include <ff/farm.hpp>


namespace grppi {

namespace detail_ff {

/**
 \brief Ordered stream reduce for FastFlow.
 */
template <typename Item, typename Reducer, typename CombineOp>
class ordered_stream_reduce : public ff::ff_ofarm {
public:
  ordered_stream_reduce(Reducer && red_obj, int num_workers);

private:
  std::vector<ff_node*> workers_;

  using emitter_type = reduce_emitter<Item,Reducer>;
  std::unique_ptr<emitter_type> p_emitter_;
};

template <typename Item, typename Reducer, typename Combiner>
ordered_stream_reduce<Item,Reducer,Combiner>::ordered_stream_reduce(
    Reducer && red_obj, 
    int num_workers) 
:
    ff::ff_ofarm{false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, num_workers},
    workers_{},
    p_emitter_{std::make_unique<emitter_type>(red_obj.window_size(), red_obj.offset())} 
{
  for(int i=0; i<num_workers; ++i) {
    reduce_worker<Item,Combiner> * p_worker =
        new reduce_worker<Item,Combiner>{red_obj.combiner()};
    workers_.push_back(p_worker);
  }

  add_workers(workers_);
  setEmitterF(p_emitter_.get());
}


} // namespace detail_ff

} // namespace grppi

#endif
