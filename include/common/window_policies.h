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
\brief Representation of count based window policy.
Represents a count based. 
It generates a window when a given number of items has arrived
\tparam ItemType type of the incoming items.
*/
template <typename ItemType>
class count_based{
public:
  using item_type = ItemType;
  /**
  \brief Constructs count based window policy.
  \param n Window size.
  \param t Sliding factor.
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
  \brief Constructs time based window policy.
  \param n Time range size of the windows.
  \param t Sliding factor.
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

template <typename ItemType>
class punctuation_based{
public:
  using item_type = ItemType;
  /**
  \brief Constructs pucntuation based window policy.
  \param n Puntutation item.
  */
  punctuation_based(item_type n) noexcept :
    punctuation_{n}
  {}

  auto add_item(ItemType && item) noexcept{
    items.push_back(item);
    return (item == punctuation_);
  }

  auto get_window() noexcept{
    auto aux{items};
    items.clear();
    return std::move(aux);
  }

private:
  item_type punctuation_;
  std::vector<ItemType> items{};
};

template <typename ItemType>
class delta_based{
public:
  using item_type = ItemType;
  /**
  \brief Constructs pucntuation based window policy.
  \param n Puntutation item.
  */
  delta_based(item_type n, item_type t, item_type initial_value) noexcept :
    delta_{n}, slide_{t}, current_value_{initial_value}
  {}


  auto add_item(ItemType && item) noexcept{
    if(item >= current_value_)
      items.push_back(item);
    return (item >= current_value_+delta_);
  }

  auto get_window() noexcept{
    auto aux{items};
    current_value_ += slide_;
    auto it = find_if(items.begin(), items.end(), 
       [&](item_type i){  return i >= current_value_ ; } );
    
    if(it!=items.end()){
      items.erase(items.begin(), it);
    }else {
      items.clear();
    }

    return std::move(aux);
  }

private:
  item_type current_value_;
  item_type delta_;
  item_type slide_;
  std::vector<ItemType> items{};
};


}
#endif
