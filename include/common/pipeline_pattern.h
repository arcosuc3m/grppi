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
#ifndef GRPPI_COMMON_PIPELINE_PATTERN_H
#define GRPPI_COMMON_PIPELINE_PATTERN_H

#include <type_traits>
#include "callable_traits.h"
namespace grppi {

/**
\brief Representation of pipeline pattern.
Represents a pipeline with multiple chained transformers.
\tparam Transformer Callable type for the first transformer.
\tparam OtherTransformers Callable types for the rest of transformers.
*/
template <typename ... Transformers>
class pipeline_t {
public:

  using transformers_type = std::tuple<Transformers...>;
  
  /**
  \brief Constructs a pipeline with several transformers.
  \param t Transformer for the first stage of the pipeline.
  \param others Rest of transformers for the other stages of the pipeline.
  */
  pipeline_t(Transformers && ... others) noexcept :
    transformers_{others...}
  {}

  pipeline_t(std::tuple<Transformers...> functions) noexcept:
    transformers_{functions}
  {}
  /**
  \brief Invokes a trasnformer from the pipeline.
  \tparam I
  */
  template <std::size_t I, typename T>
  auto invoke(T && item) const {
    auto f = std::get<I>(transformers_);
    return f(std::forward<T>(item));
  }

  /**
  \brief Gets a transformer from the pipeline
  \tparam I index into the pipeline.
  \return The selected transformer object.
  */
  template <std::size_t I>
  auto stage() const noexcept {
    static_assert(I<sizeof...(Transformers),
      "Pipeline has not so many transformers");
    return std::get<I>(transformers_);
  }

  auto transformers() const noexcept {
    return transformers_;
  }

private:
  std::tuple<Transformers...> transformers_;
};

namespace internal {

template <typename, template <typename ...> class>
struct is_pipeline : std::false_type{};

template <class... T, template <class...> class W>
struct is_pipeline <W<T...>, W> :std::true_type{};

/*
template<typename T>
struct is_pipeline : std::false_type {};

template <typename ... T>
struct is_pipeline<pipeline_t<T...>> :std::true_type {};
*/
} // namespace internal

template <class T>
static constexpr bool is_pipeline = internal::is_pipeline<std::decay_t<T>, pipeline_t>();

template <class T>
using requires_pipeline = typename std::enable_if_t<is_pipeline<T>, int>;

/*template <typename T>
static constexpr bool is_pipeline = internal::is_pipeline<std::decay_t<T>>();

template <typename T>
using requires_pipeline = typename std::enable_if_t<is_pipeline<T>, int>;
*/
namespace internal {

template <typename I, typename T>
struct output_value_type {
  using type = std::decay_t<typename std::result_of<T(I)>::type>;
};

template <typename I, typename T, typename ... U>
struct output_value_type<I,pipeline_t<T,U...>> {
  using first_result = std::decay_t<typename std::result_of<T(I)>::type>;
  using type = std::conditional_t<sizeof...(U)==0,
    first_result,
    typename output_value_type<first_result,U...>::type
  >;
};

}

template <typename I, typename T>
using output_value_type = typename internal::output_value_type<I,T>::type;

}

#endif
