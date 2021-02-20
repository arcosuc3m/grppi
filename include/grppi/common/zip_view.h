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
  template<typename ... Rs>
  class zip_view {
  public:

    /**
    \brief Construct from references to ranges.
    */
    zip_view(Rs & ... rs) // NOLINT
        : ranges_{rs...} {}

    /**
    \brief Get a tuple with the begin() of each range.
    */
    auto begin() noexcept
    {
      return begin_impl(std::make_index_sequence<sizeof...(Rs)>{});
    }

    /**
    \brief Get a tuple with the size() of each range.
    */
    auto size() const noexcept
    {
      return std::get<0>(ranges_).size();
    }

  private:
    /// Tuple of references to ranges.
    std::tuple<Rs & ...> ranges_;

  private:

    /**
    \brief Implementation details of begin()
    */
    template<std::size_t ... I>
    auto begin_impl(std::index_sequence<I...>)
    {
      return std::make_tuple(std::get<I>(ranges_).begin()...);
    }

  };

/**
\brief Factory function for easy creation of a zip_view.
\tparam Rs Ranges types.
\param rs References to ranges values.
*/
  template<typename ... Rs,
      meta::requires_<range_concept, Rs...> = 0> // NOLINT

  auto zip(Rs & ... rs)
  {
    return zip_view<Rs...>(rs...);
  }

/**
\brief A view over multiple arrays.
\tparam Rs Arrays element types.
\tparam N Size of arrays.
A view over multiple arrays that keeps references to them.
*/
  template<std::size_t N, typename ... Rs>
  class zip_view_arrays {
  public:

    /**
    \brief Construct from references to arrays.
    */
    zip_view_arrays(Rs (& ... rs)[N]) // NOLINT
        : arrays_{rs...} {}

    /**
    \brief Get a tuple with the begin() of each array.
    */
    auto begin() noexcept
    {
      return begin_impl(std::make_index_sequence<sizeof...(Rs)>{});
    }

    /**
    \brief Get a tuple with the size() of first array.
    */
    auto size() const noexcept
    {
      return N;
    }

  private:
    /// Tuple of references to arrays.
    std::tuple<Rs(&)[N]...> arrays_;

  private:

    /**
    \brief Implementation details of begin()
    */
    template<std::size_t ... I>
    auto begin_impl(std::index_sequence<I...>)
    {
      return std::make_tuple(std::begin(std::get<I>(arrays_))...);
    }

  };


/**
\brief Factory function for easy creation of a zip_view_array.
\tparam Ts Element types of arrays.
\param arrays References to ranges values.
*/
  template<typename ... Ts, std::size_t N>
  auto zip(Ts (& ...arrays)[N])
  {
    return zip_view_arrays<N, Ts...>{arrays...};
  }

}

#endif
