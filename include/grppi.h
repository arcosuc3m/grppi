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
#include "context.h"
#include "farm.h"
#include "pipeline.h"
#include "stream_filter.h"
#include "stream_iteration.h"
#include "stream_reduce.h"
#include "stream_pool.h"

namespace grppi {

/** 
\defgroup stream_patterns Streaming patterns
\brief Patterns for data stream processing.
*/

}

#endif
