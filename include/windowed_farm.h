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

#ifndef GRPPI_WINDOWED_FARM_H
#define GRPPI_WINDOWED_FARM_H

#include "common/windowed_farm_pattern.h"

namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup windowed_farm_pattern Windowed Farm pattern
\brief Interface for applyinng the \ref md_windowed_farm.
@{
*/

/**
\brief Invoke \ref md_windowed_farm on a data stream 
that can be composed in other streaming patterns.
\tparam Transformer Callable type for the transformation operation.
\tparam Window Type for the window policy.
\param ntask Concurrency degree.
\param transform_op Transformer operation.
\param window_policy policy to generate the windows.
*/
template <typename Transformer, typename Window>
auto farm(int ntasks, Transformer && transform_op, Window && window_policy)
{
   return window_farm_t<Transformer,Window>{ntasks,
       std::forward<Transformer>(transform_op),
       std::forward<Window>(window_policy)};
}

/**
@}
@}
*/

}

#endif
