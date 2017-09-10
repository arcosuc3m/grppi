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

#ifndef GRPPI_MIXED_EXECUTION_H
#define GRPPI_MIXED_EXECUTION_H

#include "common/execution_context.h"

#include <utility>


namespace grppi {

template <typename Execution, typename ... Transformers,
          template <typename...> class Pipeline,
          requires_execution_supported<Execution> = 0,
          requires_pipeline<Pipeline<Transformers...>> = 0>
auto run_with(
    const Execution & ex, 
    Pipeline<Transformers...> & pipeline_obj) 
{
  using pipe_type = Pipeline<Transformers...>;
  using context_type = execution_context_t<Execution,pipe_type>;
  return context_type(ex, pipeline_obj);
}

template <typename Execution, typename ... Transformers,
          template <typename...> class Pipeline,
          requires_execution_supported<Execution> = 0,
          requires_pipeline<Pipeline<Transformers...>> = 0>
auto run_with(
    const Execution & ex, 
    Pipeline<Transformers...> && pipeline_obj) 
{
  using pipe_type = Pipeline<Transformers...>;
  using context_type = execution_context_t<Execution,pipe_type>;
  return context_type(ex, std::forward<pipe_type>(pipeline_obj));
}

/**
@}
@}
*/

}

#endif
