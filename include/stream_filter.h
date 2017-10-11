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

#ifndef GRPPI_STREAM_FILTER_H
#define GRPPI_STREAM_FILTER_H

#include "common/patterns.h"

namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup filter_pattern Stream filter pattern
\brief Interface for applyinng the \ref md_stream-filter.
@{
*/

/**
\brief Invoke \ref md_stream-filter on a data stream
that can be composed in other streaming patterns.
This function keeps in the stream only those items
that satisfy the predicate.
\tparam Execution Execution policy.
\tparam Predicate Callable type for filter predicate.
\param ex Execution policy object.
\param predicate_op Predicate callable object.
*/
template <typename Predicate>
auto keep(Predicate && predicate_op)
{
  return filter_t<Predicate>{std::forward<Predicate>(predicate_op)};
}

/**
\brief Invoke \ref md_stream-filter on a data stream
that can be composed in other streaming patterns.
This function discards from the stream those items
that satisfy the predicate.
\tparam Execution Execution policy.
\tparam Predicate Callable type for filter predicate.
\param ex Execution policy object.
\param predicate_op Predicate callable object.
*/
template <typename Predicate>
auto discard(Predicate && predicate_op)
{
  return keep([&](auto val) { return !predicate_op(val); });
}

/**
@}
@}
*/

}

#endif
