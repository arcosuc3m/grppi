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
#ifndef GRPPI_FF_DETAIL_ORDERED_STREAM_FILTER_H
#define GRPPI_FF_DETAIL_ORDERED_STREAM_FILTER_H

#include "filter_nodes.h"

#include <ff/farm.hpp>

namespace grppi {

namespace detail_ff {

template <typename Item, typename Filter>
class ordered_stream_filter : public ff::ff_ofarm {
public:
  ordered_stream_filter(Filter && filter, int num_workers);

private:
  Filter filter_;
  std::vector<ff::ff_node *> workers_;
  std::unique_ptr<filter_collector<Item>> p_collector_;
};

template <typename Item, typename Filter>
ordered_stream_filter<Item,Filter>::ordered_stream_filter(
        Filter && filter, 
        int num_workers) 
:
    ff::ff_ofarm{false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, num_workers},
    filter_{std::move(filter)},
    workers_{},
    p_collector_{std::make_unique<filter_collector<Item>>()}
{
  for(int i=0;i<num_workers;i++) {
    workers_.push_back(new filter_worker<Item,Filter>{
        std::forward<Filter>(filter)});
  }
  add_workers(workers_);
  setCollectorF(p_collector_.get());
}


} // namespace detail_ff

} // namespace grppi

#endif
