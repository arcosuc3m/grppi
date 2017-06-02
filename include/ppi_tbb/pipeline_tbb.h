/**
* @version		GrPPI v0.1
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

#ifndef GRPPI_PIPELINE_TBB_H
#define GRPPI_PIPELINE_TBB_H

#include <tbb/pipeline.h>
#include <tbb/tbb.h>

using namespace std;
namespace grppi{
//Last stage
template <typename Stream, typename Stage>
inline const tbb::interface6::filter_t<Stream, void> stages(parallel_execution_tbb p, Stream st, Stage s ) {
    return tbb::make_filter<Stream, void>( tbb::filter::serial_in_order, s );
}

//Intermediate stages
template <typename Task, template<typename, typename> class Stage, typename Stream, typename ... Stages>
inline const tbb::interface6::filter_t<Stream, void> 
stages(parallel_execution_tbb p, Stream st, Stage<parallel_execution_tbb, Task> se, Stages ... sgs ) {
    typedef typename std::result_of<Task(Stream)>::type outputType;
    outputType k;
    return tbb::make_filter<Stream, outputType>( tbb::filter::parallel, (*se.task) ) & stages(p, k, sgs ... );
}

template <typename Stage, typename Stream, typename ... Stages>
inline const tbb::interface6::filter_t<Stream, void> 
stages(parallel_execution_tbb p, Stream st, Stage se, Stages ... sgs ) {
    typedef typename std::result_of<Stage(Stream)>::type outputType;
    outputType k;
    return tbb::make_filter<Stream, outputType>( tbb::filter::serial_in_order, se ) & stages(p, k, sgs ... );
}

//First stage
template <typename FuncIn, typename = typename std::result_of<FuncIn()>::type,
          typename ...Stages,
          requires_no_arguments<FuncIn> = 0>
void pipeline(parallel_execution_tbb p, FuncIn in, Stages ... sts ) {
    typedef typename std::result_of<FuncIn()>::type::value_type outputType;
    outputType k;
    const auto stage = tbb::make_filter<void, outputType>(
        tbb::filter::serial_in_order, 
        [&]( tbb::flow_control& fc ) {
            auto k =  in();
            if( !k ) 
                fc.stop();
            return k.value();
        }
    );
    tbb::task_group_context context;
    tbb::parallel_pipeline(p.num_tokens, stage & stages(p, k, sts ... ) );
}
}
#endif
