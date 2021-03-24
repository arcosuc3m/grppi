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
#ifndef GRPPI_COMMON_PIPELINE_PATTERN_H
#define GRPPI_COMMON_PIPELINE_PATTERN_H

#include <type_traits>

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



  template <std::size_t index, typename T, typename std::enable_if< index == (sizeof...(Transformers) -1) ,int>::type = 0 >
  auto invoke_all(T item) const
  {
      return invoke<index>(item);
  }

  template <std::size_t index, typename T, typename std::enable_if< index != (sizeof...(Transformers) -1) ,int>::type = 0 >
  auto invoke_all(T item) const
  {
      return invoke_all<index+1>(invoke<index>(item));
  }
 
  template <typename T>
  auto operator()(T item) const{
    return invoke_all<0>(item);
  }

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

  auto transformers() const & noexcept {
    return transformers_;
  }

  auto && transformers() && noexcept {
    return transformers_;
  }

  constexpr static std::size_t size() {
    return sizeof...(Transformers);
  }

  static constexpr std::size_t sizex = sizeof...(Transformers);

private:
  std::tuple<Transformers...> transformers_;
};

namespace internal {

template<typename T>
struct is_pipeline : std::false_type {};

template <typename ... T>
struct is_pipeline<pipeline_t<T...>> :std::true_type {};

} // namespace internal

template <typename T>
static constexpr bool is_pipeline = internal::is_pipeline<std::decay_t<T>>();

template <typename T>
using requires_pipeline = typename std::enable_if_t<is_pipeline<T>, int>;

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
