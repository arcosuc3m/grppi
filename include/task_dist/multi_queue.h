/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at bb
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GRPPI_MULTI_QUEUE_H
#define GRPPI_MULTI_QUEUE_H

#include <set>
#include <iostream>
#include <sstream>
#include <atomic>
#include <map>

#include "../common/mpmc_queue.h"

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

/**
  \brief Queue for storing elements from multiple servers.
  This type defines a multiple queue that can be accessed
  individually o as a whole,

  This type is a queue for storing elements from multiple servers.
*/

namespace detail {

    /**
    \brief Check if the label is contained on the set.

    Check if the label is contained on the set.
    */
    template <typename TLabel>
    bool label_in_sets(TLabel label, std::set<TLabel> set)
    {
        return (set.find(label) != set.end());
    }
    
    /**
    \brief Check if the label is contained on all sets.

    Check if the label is contained on all sets.
    */
    template <typename TLabel, typename ... TSet>
    bool label_in_sets(TLabel label, std::set<TLabel> set, TSet... sets)
    {
        return ((set.find(label) != set.end()) && label_in_sets(label, sets...));
    }
}
    
template <typename TLabel, typename TElem>
class multi_queue{
  public:

    /**
    \brief Construct an empty multi_queue.

    Creates a multi_queue with the default queue elements of a certain size.
    */
    multi_queue(long size): queue_size_{size}, label_queue_size_{size * 2}, total_used_{0}, total_occupy_{0}, queues_{}, used_{}, label_queue_{(int)label_queue_size_}
    {
      COUT << "multi_queue::multi_queue  queue_size_ = " << queue_size_ << ENDL;
    };

    /**
    \brief Registry a new label on the multi_queue.

    Registry a new label on the multi_queue, creating its label queue and used count.
    */
    void registry (TLabel queue_label) {
    try {
      COUT << "multi_queue::registry  queue_size_ = " << queue_size_ << ENDL;
      queues_.emplace(queue_label,queue_size_);
      used_.emplace(queue_label,0);
      return;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::registry ERROR: " << e.what() << std::endl;
      return;
    }
    }

    /**
    \brief Check if a label queue is registered.

    Check if a label queue is regeistered .
    */
    bool is_registered (TLabel queue_label) {
    try {
      return queues_.find(queue_label) != queues_.end();
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::is_registered(label) ERROR: " << e.what() << std::endl;
      return false;
    }
    }

    /**
    \brief Check if a label queue is empty.

    Check if a label queue is empty for an specified label.
    */
    bool empty (TLabel queue_label) {
    try {
      return queues_.at(queue_label).empty();
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::empty(label) ERROR: " << e.what() << std::endl;
      return true;
    }
    }

    /**
    \brief Check if the whole multy queue is empty.

    Check if the whole multy queue is empty.
    */
    bool empty () {
      COUT << "multi_queue::empty total_occupy_ = " << total_occupy_ << ENDL;
      return total_occupy_ <= 0;
    }

    /**
    \brief return the number of elements in the whole queue.
    \return number of elements on the whole queue.

    return the number of elements in the whole queue.
    */
    long count () {
      COUT << "multi_queue::count total_occupy_ = " << total_occupy_ << ENDL;
      return total_occupy_;
    }

    /**
    \brief Check which label queues are loaded and their labels are on the sets.

    Check which label queues are loaded and
    their label are included in all the "set" arguments
    */
    template <typename ... TSet>
    std::set<TLabel> loaded_set (TSet... sets) {
    try {
      COUT << "multi_queue::loaded_set(sets) BEGIN" << ENDL;
      COUT << "multi_queue::loaded_set(sets) queues_.size()=" << queues_.size() << ENDL;
      std::set<TLabel> ret_set;
      for (auto it=queues_.begin(); it!=queues_.end(); it++) {
          //COUT << "multi_queue::loaded_set(sets) iteracion" << ENDL;
          if (!it->second.empty() && label_in_sets(it->first, sets...)) {
            ret_set.insert(it->first);
          }
      }
      COUT << "multi_queue::loaded_set(sets) END" << ENDL;
      return ret_set;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::loaded_set(sets) ERROR: " << e.what() << std::endl;
      return {};
    }
    }

    /**
    \brief Check which label queues are loaded.

    Check which label queues are loaded
    */
    std::set<TLabel> loaded_set () {
    try {
      COUT << "multi_queue::loaded_set() BEGIN" << ENDL;
      std::set<TLabel> ret_set;
      for (auto it=queues_.begin(); it!=queues_.end(); it++) {
          if (!it->second.empty()) {
            ret_set.insert(it->first);
          }
      }
      return ret_set;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::loaded_set() ERROR: " << e.what() << std::endl;
      return {};
    }
    }
    /**
    \brief Push a new element for a certain label.
    
    Push a new element for a certain label.
    */
    template <typename TElem_FW>
    void push(TLabel queue_label, TElem_FW && elem)
    {
    try {
      COUT << "multi_queue::push() begin" << ENDL;
      if ( (total_occupy_+total_used_) >= label_queue_size_) {
        clean_label_queue();
      }
      COUT << "multi_queue::push() push queues_" << ENDL;
      queues_.at(queue_label).push(std::forward<TElem_FW>(elem));
      COUT << "multi_queue::push() push label_queue_" << ENDL;
      label_queue_.push(queue_label);
      total_occupy_++;
      COUT << "multi_queue::push() end" << ENDL;
      return;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::push() ERROR: " << e.what() << std::endl;
      return;
    }
    }

    /**
    \brief Pop a new element from any label.

    Pop a new element from any label.
    */
    TElem pop()
    {
    try {
      COUT << "multi_queue::pop() begin" << ENDL;
      auto label = label_queue_.pop();
      while (used_[label] > 0) {
        used_[label]--;
        total_used_--;
        label = label_queue_.pop();
      }
      total_occupy_--;
      auto res = queues_.at(label).pop();
      COUT << "multi_queue::pop() end" << ENDL;
      return res;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::pop() ERROR: " << e.what() << std::endl;
      return {};
    }
    }
    
    /**
    \brief Pop a new element from a certain label.

    Pop a new element from a certain label.
    */
    TElem pop(TLabel queue_label)
    {
    try {
      COUT << "multi_queue::pop(label) begin" << ENDL;
      auto elem = queues_.at(queue_label).pop();
      used_[queue_label]++;
      total_used_++;
      total_occupy_--;
      COUT << "multi_queue::pop(label) end" << ENDL;
      return elem;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue::pop(label) ERROR: " << e.what() << std::endl;
      return {};
    }
    }

  private:
    long queue_size_{0};
    long label_queue_size_{0};
    long total_used_{0};
    long total_occupy_{0};
    std::map<TLabel,locked_mpmc_queue<TElem>> queues_;
    std::map<TLabel,long> used_;
    locked_mpmc_queue<TLabel> label_queue_;

    /**
    \brief Check if the label is contained on all sets.

    Check if the label is contained on all sets.
    */
    template <typename ... TSet>
    bool label_in_sets(TLabel label, std::set<TLabel> set, TSet... sets)
    {
        return detail::label_in_sets(label, set, sets...);
    }

    /**
    \brief clean label queue.

    */
    void clean_label_queue()
    {
      COUT << "multi_queue::clean_label_queue: begin " << ENDL;
      COUT << "multi_queue::clean_label_queue total_occupy_ = " << total_occupy_ << ENDL;
      COUT << "multi_queue::clean_label_queue total_used_ = " << total_used_ << ENDL;
      long total_elem = total_occupy_ + total_used_;
      for (long i=0; i<total_elem; i++) {
        auto label = label_queue_.pop();
        if (used_[label] > 0) {
            used_[label]--;
            total_used_--;
            COUT << "multi_queue::clean_label_queue: label_queue_:  INVALID(" << i << ")" << ENDL;
        } else {
            COUT << "multi_queue::clean_label_queue: label_queue_:  VALID(" << i << ")" << ENDL;
            label_queue_.push(label);
        }
      }
      COUT << "multi_queue::clean_label_queue clean_label_queue: end " << ENDL;
      COUT << "multi_queue::clean_label_queue total_occupy_ = " << total_occupy_ << ENDL;
      COUT << "multi_queue::clean_label_queue total_used_ = " << total_used_ << ENDL;
    }
};

}

#endif

