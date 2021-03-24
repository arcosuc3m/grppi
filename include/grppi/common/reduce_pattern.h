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

  using input_type = Identity;
  using result_type = std::invoke_result_t<Combiner,Identity,Identity>;

  /**
  \brief Construct a reduction pattern object.
  \param wsize Window size.
  \param offset Offset between window starts.
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

  void add_item(const Identity & item) {
    if (remaining>0) {
      remaining--;
    }
    else {
      items.push_back(item);
    }
  }

  /**
  \pre items.size() < static_cast<int>(std::numeric_limits<int>::max())
  \brief Check if a reduction can be performed.
  */
  bool reduction_needed() const {
    return !items.empty() && (static_cast<int>(items.size()) >= window_size_);
  }

  /**
  \brief Get the combiner.
  \return The combiner held by the reduction object.
  */
  Combiner combiner() const { return combiner_; }

  /**
  \brief Get the window size.
  \return The window size of the reduction object.
  */
  int window_size() const { return window_size_; }

  /**
  \brief Get the offset.
  \return The offset the reduction object.
  */
  int offset() const { return offset_; }

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
  
  template<typename T>
  auto operator()(T &&){
    return Identity{};
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
