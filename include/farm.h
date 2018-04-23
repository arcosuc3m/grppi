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
#ifndef GRPPI_FARM_H
#define GRPPI_FARM_H

#include "common/farm_pattern.h"

namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup farm_pattern Farm pattern
\brief Interface for applyinng the \ref md_farm.
@{
*/

/**
\brief Invoke \ref md_farm on a data stream 
that can be composed in other streaming patterns.
\tparam Execution Execution policy type.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param transform_op Transformer operation.
*/
template <typename Transformer>
auto farm(int ntasks, Transformer && transform_op)
{
   return farm_t<Transformer>{ntasks,
       std::forward<Transformer>(transform_op)};
}

/**
@}
@}
*/

}

#endif
