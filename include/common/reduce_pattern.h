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

#ifndef GRPPI_COMMON_REDUCE_PATTERN_H
#define GRPPI_COMMON_REDUCE_PATTERN_H


namespace grppi{

/**
\brief Representation of reduce pattern.
Represents a reduction that can be used as a stage on a pipeline.
\tparam Combiner Callable type for the combine operation used in the reduction.
\tparam Identity Identity value for the combiner.
*/
template <typename Combiner, typename Identity>
class reduce_t {
public:

  /**
  \brief Construct a reduction pattern object.
  \param wsize Window size.
  \param offset Offset betwee window starts.
  \param Id Identity value.
  \param combine_op Combiner used for the reduction.
  */
  reduce_t(int wsize, int offset, Identity id, Combiner && combine_op) :
    window_size_{wsize}, offset_{offset}, 
    identity_{id}, combiner_{combine_op}
  {}

  /**
  \brief Add an item to the reduction buffer.
  If there are remaining items before reaching the next window start the
  item is discarded.
  \param item to be added.
  */
  void add_item(Identity && item) {
    if (remaining>0) {
      remaining--;
    }
    else {
      items.push_back(std::forward<Identity>(item));
    }
  }

  /**
  \brief Check if a reduction can be performed.
  */
  bool reduction_needed() const {
    return !items.empty() && (items.size() >= window_size_);
  }

  /**
  \brief Reduce values from a window.
  \return The result of the reduction.
  */
  template <typename E>
  auto reduce_window(const E & e) {
    auto red = e.reduce(items.begin(), items.size(), identity_, combiner_);
    if (offset_ > window_size_) {
      remaining = offset_ - window_size_;
      items.clear();
    }
    else {
      items.erase(items.begin(), std::next(items.begin(), offset_));
    }
    return red;
  }

private:
  int window_size_;
  int offset_;
  Identity identity_;
  Combiner combiner_;

  std::vector<Identity> items{};
  int remaining = 0;
};

namespace internal {

template<typename T>
struct is_reduce : std::false_type {};

template <typename C, typename I>
struct is_reduce<reduce_t<C,I>> :std::true_type {};

}

template <typename T>
constexpr bool is_reduce = internal::is_reduce<std::decay_t<T>>();

template <typename T>
using requires_reduce = std::enable_if_t<is_reduce<T>,int>;

} // end namespace grppi

#endif
