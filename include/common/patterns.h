/**
* @version		GrPPI v0.2
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

#ifndef GRPPI_COMMON_PATTERNS_H
#define GRPPI_COMMON_PATTERNS_H

#include <tuple> 

#include "farm_pattern.h"
#include "filter_pattern.h"
#include "pipeline_pattern.h"
#include "reduce_pattern.h"
#include "iteration_pattern.h"

namespace grppi{

template <typename,typename...> struct get_return_type;


template <typename Input, typename Stage>
struct get_return_type<Input,Stage>
{
  using type = typename std::result_of<Stage(Input)>::type;
};

template <typename Input, typename Stage, typename ...Stages> 
struct get_return_type<Input, Stage, Stages ...>
{
  using type = typename get_return_type<typename std::result_of<Stage(Input)>::type, Stages...>::type;
};

template <typename E,typename Stage, typename ... Stages>
class pipeline_info{
   public:
      E & exectype;
      std::tuple<Stage , Stages ...> stages;
      pipeline_info(E &p, Stage s, Stages ... sts) : exectype{p}, stages{std::make_tuple(s, sts...)} {}
      pipeline_info(E &p, std::tuple<Stage,Stages ...> st) : exectype{p} , stages{st} {}
};

template <typename E,class Combiner, typename Identity>
class reduction_info
{
   public:
      Combiner combine_op;
      int window_size;
      int offset;
      Identity identity;
      E & exectype;
      reduction_info(E &s,int ws, int off, Identity iden, Combiner comb) : 
        exectype{s}, window_size{ws}, offset{off}, identity{iden}, combine_op{comb} {}
};

template <typename E,class Operation>
class farm_info
{
   public:
      Operation  task;
      E & exectype;
      int farmtype;
      farm_info(E &s,Operation  f) : task{f}, exectype{s}, farmtype{} {};
};

template <typename E,class Operation>
class filter_info
{
   public:
      Operation task;
      E & exectype;
      int filtertype;
      filter_info(E &s,Operation f) : task{f}, exectype{s}, filtertype{} {};
};

template <typename T>
constexpr bool is_no_pattern =
  !is_farm<T> && 
  !is_filter<T> && 
  !is_pipeline<T> &&
  !is_reduce<T> &&
  !is_iteration<T>;


template <typename T>
constexpr bool is_pattern = !is_no_pattern<T>;

template <typename T>
using requires_no_pattern = std::enable_if_t<is_no_pattern<T>,int>;

template <typename T>
using requires_pattern = std::enable_if_t<is_pattern<T>, int>;

//Base template to determine the input type of a function or pattern
template<class T, class Enable = void>
class input_type {}; // primary template
 
//Specialization for patterns
template<class T>
class input_type<T, typename std::enable_if<  !is_no_pattern<T> >::type> {
  public:
  using type = typename T::input_type;
};

//Specialization for fuctions
template<class T>
class input_type<T, typename std::enable_if<  is_no_pattern<T> >::type> {
  public:
  using type = typename internal::function_traits<T>::template arg<0>::type;
};

//Base template to take the first argument of a function or a pattern
template <typename T>
struct next_type{
  using type = typename input_type<typename std::decay<T>::type>::type;
};

//Variadic template used for multiple functions/patterns
template <typename T, typename ... Other>
struct next_input_type{
  using type = typename next_type<T>::type;
};


} // end namespace grppi

#endif
