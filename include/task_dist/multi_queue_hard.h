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
#ifndef GRPPI_MULTI_QUEUE_HARD_H
#define GRPPI_MULTI_QUEUE_HARD_H

#include <set>
#include <iostream>
#include <sstream>
#include <atomic>
#include <map>
#include <utility>
#include <experimental/optional>
#include <memory>
#include <tuple>
#include <cassert>
#include <exception>


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

namespace detail_hard {

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
class multi_queue_hard{
  public:
    
    /**
    \brief Construct an empty multi_queue_hard.

    Creates a multi_queue_hard with the default queue elements of a certain size.
    */
    multi_queue_hard(long size): queue_size_{size}, label_queue_size_{size * 2}, total_used_{0}, total_strict_{0}, total_valid_{0}, total_occupy_{0}, queues_{}, label_queue_{(int)label_queue_size_}, used_{}, strict_{}, valid_{}
    {
      COUT << "multi_queue_hard::multi_queue_hard  queue_size_ = " << queue_size_ << ENDL;
    };

    /**
    \brief Registry a new label on the multi_queue_hard.

    Registry a new label on the multi_queue_hard, creating its label queue and used count.
    */
    void registry (TLabel queue_label) {
    try {
      COUT << "multi_queue_hard::registry  queue_size_ = " << queue_size_ << ENDL;
      queues_.emplace(queue_label,queue_size_);
      used_.emplace(queue_label,0);
      strict_.emplace(queue_label,0);
      return;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue_hard::registry ERROR: " << e.what() << std::endl;
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
      std::cerr << "multi_queue_hard::is_registered(label) ERROR: " << e.what() << std::endl;
      return false;
    }
    }
    
    /**
    \brief Check if a label queue is empty.

    Check if a label queue is empty for an specified label.
    */
    bool empty (TLabel queue_label) {
    try {
      return valid_[queue_label]==0;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue_hard::empty(label) ERROR: " << e.what() << std::endl;
      return true;
    }
    }

    /**
    \brief Check if the whole multy queue is empty.

    Check if the whole multy queue is empty.
    */
    bool empty_all () {
      COUT << "multi_queue_hard::empty_all total_occupy_ = " << total_occupy_ << ENDL;
      return total_occupy_ <= 0;
    }

    /**
    \brief Check if the whole multy queue is empty of non strict elements.

    Check if the whole multy queue is empty of non strict elements.
    */
    bool empty () {
      COUT << "multi_queue_hard::empty total_occupy_ = " << total_occupy_ << ", total_strict_ = " << total_strict_ << ENDL;
      return total_occupy_-total_strict_ <= 0;
    }


    /**
    \brief return the number of elements in the whole queue.
    \return number of elements on the whole queue.

    return the number of elements in the whole queue.
    */
    long count () {
      COUT << "multi_queue_hard::count total_occupy_ = " << total_occupy_ << ENDL;
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
      std::set<TLabel> ret_set;
      for (auto it=queues_.begin(); it!=queues_.end(); it++) {
          if ((valid_[it->first]!=0) && label_in_sets(it->first, sets...)) {
            ret_set.insert(it->first);
          }
      }
      return ret_set;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue_hard::loaded_set(sets) ERROR: " << e.what() << std::endl;
      return {};
    }
    }

    /**
    \brief Check which label queues are loaded.

    Check which label queues are loaded
    */
    std::set<TLabel> loaded_set () {
    try {
      std::set<TLabel> ret_set;
      for (auto it=queues_.begin(); it!=queues_.end(); it++) {
          if (valid_[it->first]!=0) {
            ret_set.insert(it->first);
          }
      }
      return ret_set;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue_hard::loaded_set(sets) ERROR: " << e.what() << std::endl;
      return {};
    }
    }
    
    /**
    \brief Push a new non-strict element for a certain label.
    
    Push a new non-strict element for a certain labe l.
    */
    void push(TLabel queue_label, TElem elem, bool strict_label)
    {
      return push(std::vector<TLabel>{queue_label}, elem, strict_label);
    }
    
    /**
    \brief Push a new element for a certain label (strict or non strict).
    
    Push a new element for a certain label (strict or non strict).
    */
    void push(std::vector<TLabel> list_labels, TElem elem, bool strict_label)
    {
    try {
      COUT << "multi_queue_hard::push begin" << ENDL;
      if ( ((total_occupy_+total_used_)-total_strict_) >= label_queue_size_) {
        clean_label_queue();
      }
      using packed_type = std::experimental::optional<std::tuple <std::vector<TLabel>, TElem, bool>>;
      auto packed = std::make_shared<packed_type>(
                    std::make_tuple(list_labels, elem, strict_label) );

      {
      auto list_labels = std::get<0>(packed->value());
      auto elem = std::get<1>(packed->value());
      auto strict_label = std::get<2>(packed->value());

      COUT << "multi_queue_hard::push elem =  << elem << , strict_label = " << strict_label << ", list_labels.size() = " << list_labels.size() << ", list_labels[0] = " << list_labels[0] << ENDL;
      }
      auto prime_label = *(list_labels.begin());
      for (auto it=list_labels.begin(); it!=list_labels.end(); it++) {
        queues_.at(*it).push(packed);
        if (strict_label) {
          strict_[*it]++;
        }
        valid_[*it]++;
      }
      if (strict_label) {
        total_strict_++;
      } else {
        label_queue_.push(prime_label);
      }
      total_occupy_++;
      COUT << "multi_queue_hard::push end" << ENDL;
      return;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue_hard::push() ERROR: " << e.what() << std::endl;
      return;
    }
    }

    /**
    \brief Pop a new non-strict element from any label.

    Pop a new non-strict element from any label.
    */
    TElem pop()
    {
    try {
      COUT << "multi_queue_hard::pop begin" << ENDL;
      // check there are valid elements in any queue
      assert (!empty());
      auto label = label_queue_.pop();
      while (used_[label] > 0) {
        COUT << "multi_queue_hard::pop label used take another" << ENDL;
        used_[label]--;
        total_used_--;
        label = label_queue_.pop();
      }
      total_occupy_--;
      auto packed = queues_.at(label).pop();
      while ( (!*packed) || (std::get<2>(packed->value())) ) {
        COUT << "multi_queue_hard::pop() not valid get another" << ENDL;
        if (*packed) { // is an strict_label element
          COUT << "multi_queue_hard::pop() strict_label element, push it back" << ENDL;
          queues_.at(label).push(packed);
        }
        packed = queues_.at(label).pop();
      }
      auto list_labels = std::get<0>(packed->value());
      auto elem = std::get<1>(packed->value());
      auto strict_label = std::get<2>(packed->value());
      (*packed) = {};   // reset optional
      COUT << "multi_queue_hard::pop elem = << elem << , strict_label = " << strict_label << ", list_labels.size() = " << list_labels.size() << ", list_labels[0] = " << list_labels[0] << ENDL;

      assert (!strict_label);
      for (auto it=list_labels.begin(); it!=list_labels.end(); it++) {
        valid_[*it]--;
      }
      return elem;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue_hard::pop() ERROR: " << e.what() << std::endl;
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
      COUT << "multi_queue_hard::pop(label) begin" << ENDL;
      // check there are valid elements in the queue
      assert (!empty(queue_label));
      auto packed = queues_.at(queue_label).pop();
      while (!*packed)  {
        COUT << "multi_queue_hard::pop(label) not valid get another" << ENDL;
        packed = queues_.at(queue_label).pop();
      }
      auto list_labels = std::get<0>(packed->value());
      auto elem = std::get<1>(packed->value());
      auto strict_label = std::get<2>(packed->value());
      (*packed) = {};   // reset optional
      auto prime_label = *(list_labels.begin());
      COUT << "multi_queue_hard::pop elem = << elem << , strict_label = " << strict_label << ", list_labels.size() = " << list_labels.size() << ", prime_label = " << prime_label << ENDL;

      for (auto it=list_labels.begin(); it!=list_labels.end(); it++) {
        if (strict_label) {
          strict_[*it]--;
        }
        valid_[*it]--;
      }
      if (strict_label) {
        total_strict_--;
      } else {
        used_[prime_label]++;
        total_used_++;
      }
      total_occupy_--;
      COUT << "multi_queue_hard::pop(label) end" << ENDL;
      return elem;
    } catch(const std::exception &e) {
      std::cerr << "multi_queue_hard::pop(label) ERROR: " << e.what() << std::endl;
      return {};
    }
    }

  private:
    long queue_size_{0};
    long label_queue_size_{0};
    long total_used_{0};
    long total_strict_{0};
    long total_valid_{0};
    long total_occupy_{0};
    std::map<TLabel, locked_mpmc_queue <
        std::shared_ptr <std::experimental::optional <
        std::tuple <std::vector<TLabel>, TElem, bool> >> >> queues_;
    locked_mpmc_queue<TLabel> label_queue_;
    std::map<TLabel,long> used_;
    std::map<TLabel,long> strict_;
    std::map<TLabel,long> valid_;


    /**
    \brief Check if the label is contained on all sets.

    Check if the label is contained on all sets.
    */
    template <typename ... TSet>
    bool label_in_sets(TLabel label, std::set<TLabel> set, TSet... sets)
    {
        return detail_hard::label_in_sets(label, set, sets...);
    }

    /**
    \brief clean label queue.

    */
    void clean_label_queue()
    {
      COUT << "multi_queue::clean_label_queue: begin " << ENDL;
      COUT << "multi_queue::clean_label_queue: total_occupy_ = " << total_occupy_ << ENDL;
      COUT << "multi_queue::clean_label_queue: total_used_ = " << total_used_ << ENDL;
      COUT << "multi_queue::clean_label_queue: total_strict_ = " << total_strict_ << ENDL;
      long total_elem = (total_occupy_ + total_used_) - total_strict_;
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
      COUT << "multi_queue::clean_label_queue:: end " << ENDL;
      COUT << "multi_queue::clean_label_queue: total_occupy_ = " << total_occupy_ << ENDL;
      COUT << "multi_queue::clean_label_queue: total_used_ = " << total_used_ << ENDL;
      COUT << "multi_queue::clean_label_queue: total_strict_ = " << total_strict_ << ENDL;
    }
};

}

#endif

