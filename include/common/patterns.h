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

namespace grppi{

template <typename E,typename Stage, typename ... Stages>
class pipeline_info{
   public:
      E & exectype;
      std::tuple<Stage , Stages ...> stages;
      pipeline_info(E &p, Stage s, Stages ... sts) : exectype{p}, stages{std::make_tuple(s, sts...)} {}
};

template <typename E,class Operation, class RedFunc>
class reduction_info
{
   public:
      Operation task;
      RedFunc red;
      E & exectype;
      reduction_info(E &s, Operation farm, RedFunc r) : task{farm}, red{r}, exectype{s} {}
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

} // end namespace grppi

#endif
