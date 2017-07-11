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

#ifndef GRPPI_TBB_PIPELINE_H
#define GRPPI_TBB_PIPELINE_H

#ifdef GRPPI_TBB

#include <experimental/optional>

#include <tbb/pipeline.h>
#include <tbb/tbb.h>

namespace grppi{
//Last stage
template <typename Stream, typename Stage>
 const tbb::interface6::filter_t<std::experimental::optional<Stream>, void> stages(parallel_execution_tbb &p, Stream st, Stage && se ) {
    return tbb::make_filter<std::experimental::optional<Stream>, void>( tbb::filter::serial_in_order,[&](std::experimental::optional<Stream> s){ if(s) se(s.value());} );
}


//Intermediate stages
template <typename Operation, typename Stream, typename ... Stages>
 const tbb::interface6::filter_t< std::experimental::optional<Stream>, void> 
stages(parallel_execution_tbb &p, Stream st, farm_info<parallel_execution_tbb, Operation> & se, Stages && ... sgs ) {
   return stages(p,st, std::forward< farm_info<parallel_execution_tbb, Operation> &&>( se), std::forward<Stages>( sgs )... );
}


template <typename Operation, typename Stream, typename... Stages>
const tbb::interface6::filter_t< std::experimental::optional<Stream>, void> stages(parallel_execution_tbb &p, Stream& st,
            filter_info<parallel_execution_tbb,Operation> & se, Stages && ... sgs ) {
     return stages(p,st,std::forward<filter_info<parallel_execution_tbb,Operation> &&>( se ), std::forward<Stages>( sgs )... );

}



template <typename Operation, typename Stream, typename... Stages>
const tbb::interface6::filter_t< std::experimental::optional<Stream>, void> stages(parallel_execution_tbb &p, Stream& st,
            filter_info<parallel_execution_tbb,Operation> && se, Stages && ... sgs ) {
    Stream k;
    return tbb::make_filter< std::experimental::optional<Stream>, std::experimental::optional<Stream> >( tbb::filter::parallel,
       [&](std::experimental::optional<Stream> s)
            { return (s && se.task(s.value())) ? s : std::experimental::optional<Stream>();} ) & stages(p, k, std::forward<Stages>(sgs) ... );

}


template <typename Operation, typename Stream, typename ... Stages>
 const tbb::interface6::filter_t<std::experimental::optional<Stream>, void> 
stages(parallel_execution_tbb &p, Stream st, farm_info<parallel_execution_tbb, Operation> && se, Stages && ... sgs ) {
    typedef typename std::result_of<Operation(Stream)>::type outputType;
    outputType k;
    return tbb::make_filter< std::experimental::optional<Stream>, std::experimental::optional<outputType> >( tbb::filter::parallel,
       [&](std::experimental::optional<Stream> s)
            {return (s) ? std::experimental::optional<outputType>(se.task( s.value() )) : std::experimental::optional<outputType>();} ) & stages(p, k, std::forward<Stages>(sgs) ... );
}

template <typename Stage, typename Stream, typename ... Stages>
 const tbb::interface6::filter_t<std::experimental::optional<Stream>, void> 
stages(parallel_execution_tbb &p, Stream st, Stage &&  se, Stages && ... sgs ) {
    typedef typename std::result_of<Stage(Stream)>::type outputType;
    outputType k;
    return tbb::make_filter< std::experimental::optional<Stream>, std::experimental::optional<outputType> >( tbb::filter::serial_in_order,
       [&](std::experimental::optional<Stream> s)
            {return (s) ? std::experimental::optional<outputType>( se( s.value() ) ): std::experimental::optional<outputType>();} ) & stages(p, k, std::forward<Stages>(sgs) ... );
}

//First stage
template <typename FuncIn, 
          typename ...Stages,
          requires_no_arguments<FuncIn> = 0>
void pipeline(parallel_execution_tbb &p, FuncIn in, Stages && ... sts ) {
    typedef typename std::result_of<FuncIn()>::type::value_type outputType;
    outputType k;
    const auto stage = tbb::make_filter<void, std::experimental::optional<outputType> >(
        tbb::filter::serial_in_order, 
        [&]( tbb::flow_control& fc ) {
            auto item =  in();
            if( !item ) 
                fc.stop();
            return (item) ? std::experimental::optional<outputType>{item.value()} :std::experimental::optional<outputType>{};
        }
    );
    tbb::task_group_context context;
    tbb::parallel_pipeline(p.num_tokens, stage & stages(p, k, std::forward<Stages>(sts) ... ) );
}
}
#endif

#endif
