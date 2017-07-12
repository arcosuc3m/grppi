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

#ifndef GRPPI_SEQ_PIPELINE_H
#define GRPPI_SEQ_PIPELINE_H

namespace grppi {

template <typename InType, int currentStage, typename ...Stages>
 typename std::enable_if<(currentStage == (sizeof...(Stages)-1)), InType>::type composed_pipeline(InType in, pipeline_info<sequential_execution, Stages...> const & pipe)
{
     return (*std::get<currentStage>(pipe.stages))(in);
}


template <typename InType, int currentStage, typename ...Stages>
 typename std::enable_if<(currentStage < (sizeof...(Stages)-1)), InType>::type composed_pipeline(InType in, pipeline_info<sequential_execution, Stages...> const & pipe)
{
     auto val = (*std::get<currentStage>(pipe.stages))(in);
     return composed_pipeline<InType, currentStage+1, Stages...>(val,pipe);
}



//Last stage
template <typename Stream, typename Stage>
void stages(sequential_execution &s, Stream st, Stage && se ) {
    se( st );
}




//Filter stage
template <typename task, typename Stream, typename... Stages>
void stages(sequential_execution &s, Stream st, filter_info<sequential_execution, task> & se, Stages && ... sgs ) {
      stages(s, st, std::forward<filter_info<sequential_execution, task>&&>(se), std::forward<Stages>(sgs)... );
}

template <typename task, typename Stream, typename... Stages>
void stages(sequential_execution &s, Stream st, filter_info<sequential_execution, task> && se, Stages && ... sgs ) {
     if(se.task(st))
        stages(s, st, std::forward<Stages>(sgs) ... );
}

//Farm stage
template <typename task, typename Stream, typename... Stages>
void stages(sequential_execution &s, Stream st, farm_info<sequential_execution, task> & se, Stages && ... sgs ) {
     stages(s, st, std::forward<farm_info<sequential_execution, task>&&>(se), std::forward<Stages>(sgs) ... );
}

template <typename task, typename Stream, typename... Stages>
void stages(sequential_execution &s, Stream st, farm_info<sequential_execution, task> && se, Stages && ... sgs ) {
     auto out = se.task(st);
     stages(s, out, std::forward<Stages>(sgs) ... );
}



//Intermediate stages
template <typename Stage, typename Stream, typename... Stages>
void stages(sequential_execution &s, Stream st, Stage && se, Stages && ... sgs ) {
    auto out = se( st );
    stages(s, out, std::forward<Stages>(sgs) ... );
}

//First stage
template <typename FuncIn, typename = typename std::result_of<FuncIn()>::type, typename ...Stages>
void pipeline(sequential_execution &s, FuncIn && in, Stages && ... sgs ) {

    while( 1 ) {
        auto k = in();
        if( !k )
            break;
        stages(s, k.value(), std::forward<Stages>(sgs) ... );
    }
}
}
#endif
