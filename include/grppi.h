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

#ifndef GRPPI_GRPPI_H
#define GRPPI_GRPPI_H

// Includes for execution policies
#include "dyn/dynamic_execution.h"

// Includes for data parallel patterns
#include "map.h"
#include "mapreduce.h"
#include "reduce.h"
#include "stencil.h"

namespace grppi {

/** 
\defgroup data_patterns Data parallel patterns
\brief Patterns for data parallel processing.
*/

}

// Includes for task patterns
#include "divideconquer.h"

namespace grppi {

/** 
\defgroup task_patterns Task parallel patterns
\brief Patterns for task parallel processing.
*/
}

// Includes for streaming patterns
#include "farm.h"
#include "pipeline.h"
#include "stream_filter.h"
#include "stream_iteration.h"
#include "stream_reduce.h"

namespace grppi {

/** 
\defgroup stream_patterns Streaming patterns
\brief Patterns for data stream processing.
*/

}

#endif
