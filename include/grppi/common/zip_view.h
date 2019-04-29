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
#ifndef GRPPI_COMMON_ZIP_VIEW_H
#define GRPPI_COMMON_ZIP_VIEW_H

#include "range_concept.h"

namespace grppi {

/**
\brief A view over multiple ranges.
\tparam Rs Ranges types.
A view over multiple ranges that keeps references to them.
*/
template <typename ... Rs>
class zip_view {
public:

  /**
  \brief Construct from references to ranges.
  */
  zip_view(Rs& ... rs) : rngs_{rs...} {}

  /**
  \brief Get a tuple with the begin() of each range.
  */
  auto begin() noexcept { 
    return begin_impl(std::make_index_sequence<sizeof...(Rs)>{}); 
  }

  /**
  \brief Get a tuple with the size() of each range.
  */
  auto size() const noexcept {
    return std::get<0>(rngs_).size();
  }

private:
  /// Tuple of references to ranges.
  std::tuple<Rs&...> rngs_;

private:

  /**
  \brief Implementation details of begin()
  */
  template <std::size_t ... I>
  auto begin_impl(std::index_sequence<I...>) {
    return std::make_tuple(std::get<I>(rngs_).begin()...);
  }

};

/**
\brief Factory function for easy creation of a zip_view.
\tparam Rs Ranges types.
\param rs References to ranges values.
*/
template <typename ... Rs,
          meta::requires<range_concept, Rs...> = 0>
auto zip(Rs & ... rs) { 
  return zip_view<Rs...>(rs...); 
}

}

#endif
