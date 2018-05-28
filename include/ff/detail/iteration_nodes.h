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
#ifndef GRPPI_FF_DETAIL_ITERATION_NODES_H
#define GRPPI_FF_DETAIL_ITERATION_NODES_H

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

#endif
