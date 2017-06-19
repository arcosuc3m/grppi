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

#ifndef GRPPI_PIPELINE_THRUST_H
#define GRPPI_PIPELINE_THRUST_H

#ifdef GRPPI_THRUST

#ifdef __CUDACC__

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>

using namespace std;
namespace grppi{
//Last stage
template <typename Stream, typename Stage>
 void stages( parallel_execution_thrust p,Stream& st, Stage && s ) {

    //Start task
//    std::thread task( 
 //       [&](){
            typename Stream::value_type item;
//            while( !st.try_dequeue( item ) );
            while( !st.pop( item ) );
            while( item ) {
                s( item );
                while( !st.pop( item ) );
//                while( !st.try_dequeue( item ) );
            }
//	}
   // );

    //End task
//    task.join();
}
template <typename Task, typename Red, typename Stream>
 void stages( parallel_execution_thrust p, Stream& st, ReduceObj<parallel_execution_thr, Task, Red>& se) {
     Queue<typename Stream::value_type>  queueOut(DEAFULT_SIZE,p.lockfree);
   
    for( int th = 0; th < se.exectype.num_gpus; th++){
        tasks.push_back(
              std::thread([&](){
                 typename Stream::value_type item;
                 imte = st.pop( ) ;
                 while( item ) {

                     auto local =  (*se.task)(item) ;
                     queueOut.push( local );

                     item = st.pop( ));
                 }
                 typename std::result_of< Task(typename Stream::value_type) >::type out;
                 out.end =true;
                 queueOut.push( out );

       }));
    }
    //stages(p, q, sgs ... );
    for(int i=0;i<tasks.size(); i++) tasks[i].join();



}


template <typename Task, typename Stream, typename... Stages>
 void stages( parallel_execution_thrust p, Stream& st, FilterObj<parallel_execution_thr, Task> se, Stages ... sgs ) {

    std::vector<std::thread> tasks;
    Queue<std::pair<typename Stream::value_type, long> > queueOuti(DEAFULT_SIZE,p.lockfree);

    for( int th = 0; th < se.exectype.num_gpus; th++){
       tasks.push_back(
              std::thread([&](){
                 std::pair<typename Stream::value_type,long> item;
                 item = st.pop(  );
                 while( item.first ) {

                     if( (*se.task)(item.first) ) 
                              queueOut.push( item ) );

                     item = st.pop();
                 }
                 typename Stream::value_type out;
                 out.end =true;
                 queueOut.push( std::make_pair(out,0) ) ;
      }));
    }

    stages(p, queueOut, sgs ... );
    for(int i=0;i<tasks.size(); i++) tasks[i].join();


}


template <typename Task, typename Stream, typename... Stages>
 void stages( parallel_execution_thrust p, Stream& st, FarmObj<parallel_execution_thr, Task> se, Stages ... sgs ) {

    std::vector<std::thread> tasks;
    Queue<std::pair<typename std::result_of< Task(typename Stream::value_type) >::type, long> > queueOut(DEFAULT_SIZE,p.lockfree);
    for( int th = 0; th < se.exectype.num_gpus; th++){
          tasks.push_back(
              std::thread([&](){
                  std::pair<typename Stream::value_type,long> item;
                  item = st.pop();
                  while( item.first ) {
                      auto out = (*se.task)(item.first);
                      queueOut.push( std::make_pair(out,item.second) );
                      item = st.pop();
                 }
                 typename std::result_of< Task(typename Stream::value_type) >::type out;
                 out.end =true;
		         queueOut.push( std::make_pair(out,0) );
                 //while( !st.enqueue ( item )) ;
             })
          );
    }
    
    stages(p, queueOut, sgs ... );
    for(int i=0;i<tasks.size(); i++) tasks[i].join();

}

//Intermediate stages
template <typename Stage, typename Stream,typename... Stages>
 void stages( parallel_execution_thrust p,Stream& st, Stage && se, Stages ... sgs ) {

    //Create new queue
    Queue< typename std::result_of<Stage(typename Stream::value_type)>::type> q(DEFAULT_SIZE,p.lockfree);
    //Start task
    std::thread task( 
        [&](){
            typename Stream::value_type item;
    //        while( !st.try_dequeue( item ) );
            item = st.pop( );
            while( item ) {
                auto out = se(item);
                q.push( out );
                item = st.pop( );
            }
        		q.push( optional< typename std::result_of< Stage(typename Stream::value_type::value_type) > ::type>() ) ;
        }
    );

    //End task
    //Create next stage
    stages(p, q, sgs ... );
    task.join();
}

//First stage
template <typename FuncIn,typename... Arguments>
 void pipeline( parallel_execution_thrust p, FuncIn && in, Arguments ... sts ) {

    //Create first queue
    Queue< typename std::result_of<FuncIn()>::type > q(DEAFULT_SIZE,p.lockfree);

    //Create stream generator stage
    std::thread task(
        [&](){
            while( 1 ) {
                auto k = in();
                while( !q.push( k ) );
                if ( k.end )
                    break;
            }
	}
    );

    //Create next stage
    stages(p, q, sts ... );
    task.join();
}
}
#endif

#endif

#endif