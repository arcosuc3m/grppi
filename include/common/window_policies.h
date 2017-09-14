/**
* @version		GrPPI v0.3
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
#ifndef GRPPI_COMMON_WINDOW_POLICIES_H
#define GRPPI_COMMON_WINDOW_POLICIES_H

#include <type_traits>
#include <chrono>
#include <algorithm>
namespace grppi {

/**
\brief Representation of farm pattern.
Represents a farm of n replicas from a transformer.
\tparam Transformer Callable type for the farm transformer.
*/
template <typename ItemType>
class count_based{
public:
  using item_type = ItemType;
  /**
  \brief Constructs a farm with a cardinality and a transformer.
  \param n Number of replicas for the farm.
  \param t Transformer for the farm.
  */
  count_based(int n, int t) noexcept :
    window_size_{n}, offset_{t}
  {}

  auto add_item(ItemType && item) noexcept{
    if (remaining == 0) {
      items.push_back(item);
    } 
    else {
      remaining--;
    }
    return (items.size() == window_size_);
  }
 
  auto get_window() noexcept{
    auto aux{items};
    if (offset_ >= window_size_) {
      remaining = offset_ - window_size_;
      items.clear();
    }
    else {
      items.erase(items.begin(), std::next(items.begin(), offset_));
    }
    return std::move(aux);
  }

private:
  int window_size_;
  int offset_;
 
  std::vector<ItemType> items{};
  int remaining = 0;
};

template <typename ItemType>
class time_based {
public:
  using item_type = ItemType;
  using time_type = std::chrono::high_resolution_clock::time_point;
  /**
  \brief Constructs a farm time based window.
  \param n time range size of the windows in seconds.
  \param offset overlapping time between windows in secods.
  */
  time_based(double n, double t) noexcept :
    window_size_{n}, offset_{t}
  {
    last_time = std::chrono::high_resolution_clock::now();
  }

  auto add_item(ItemType && item){
    time_type current_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - last_time);
    if ( time.count() > 0.0 )
      items.emplace_back(current_time, item);
    return time.count() > window_size_;
  }
  
  auto get_window(){
    std::vector<ItemType> aux;
    std::transform( items.begin(), items.end()-1, std::back_inserter( aux ),
      [](const std::pair<time_type, int> &p) { return p.second; } );
    
    if (offset_ > window_size_) {   
      items.clear();
    }
    else {
      auto next_slide = std::find_if(items.begin(), items.end(),
        [&](const std::pair<time_type, ItemType> &p) { 
          return std::chrono::duration_cast<std::chrono::duration<double>>(p.first - last_time).count() >= offset_; } 
      );
      items.erase(items.begin(), next_slide);
    }
    int seconds = offset_;
    int milliseconds = (offset_ * 1000) - seconds;    
    last_time = last_time + std::chrono::seconds(seconds) + std::chrono::milliseconds(milliseconds);
    return aux;
  }

private:
  double window_size_;
  double offset_;
  
  std::vector<std::pair<time_type, ItemType>> items{};

  std::chrono::high_resolution_clock::time_point last_time;
  
  
};

}

#endif
