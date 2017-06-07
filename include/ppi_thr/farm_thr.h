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

#ifndef GRPPI_FARM_THR_H
#define GRPPI_FARM_THR_H


#include <thread>
#include <utility>
#include <memory>
using namespace std;
namespace grppi{

template <typename GenFunc, typename TaskFunc, typename SinkFunc>
inline void farm(parallel_execution_thr &p, GenFunc const &in, TaskFunc const & taskf , SinkFunc const &sink) {

    std::vector<std::thread> tasks;
//    Queue< typename std::result_of<GenFunc()>::type > queue(DEFAULT_SIZE);
//    Queue< optional < typename std::result_of<TaskFunc(typename std::result_of<GenFunc()>::type::value_type)>::type > > queueout(DEFAULT_SIZE);
    Queue< typename std::result_of<GenFunc()>::type > queue (DEFAULT_SIZE,p.is_lockfree());
    Queue< optional < typename std::result_of<TaskFunc(typename std::result_of<GenFunc()>::type::value_type)>::type > > queueout(DEFAULT_SIZE, p.is_lockfree());
    std::atomic<int> nend(0);
    //Create threads
    for( int i = 0; i < p.get_num_threads(); i++ ) {
        tasks.push_back(
            std::thread(
                [&](){
                    // Register the thread in the execution model
                    p.register_thread();

                    typename std::result_of<GenFunc()>::type item;
                    item = queue.pop( ) ;
                    //auto item = queue.pop( );
                    while( item ) {
                       auto out = taskf( item.value() );
                       queueout.push( optional < typename std::result_of<TaskFunc(typename std::result_of<GenFunc()>::type::value_type)>::type >(out) );
                       // item = queue.pop( );
                       item = queue.pop( ) ;
                    }
                    queue.push(item);
                    nend++;
                    if(nend == p.get_num_threads())
                        queueout.push( optional< typename std::result_of<TaskFunc(typename std::result_of<GenFunc()>::type::value_type)>::type >() ) ;

                    // Deregister the thread in the execution model
                    p.deregister_thread();
                }
            )
        );
    }

    //SINK 
    tasks.push_back(
         std::thread(
            [&](){
                // Register the thread in the execution model
                p.register_thread();

                optional< typename std::result_of<TaskFunc(typename std::result_of<GenFunc()>::type::value_type)>::type > item;
                item = queueout.pop( ) ;
                // auto item = queueout.pop(  ) ;
                 while( item ) {
                    sink( item.value() );
//                  item = queueout.pop(  ) ;
                    item = queueout.pop( );
                 }

                // Deregister the thread in the execution model
                p.deregister_thread();
             }
        )
    );

   //Generate elements
    while( 1 ) {
        auto k = in();
        queue.push( k );
        if( !k ) {
/*            for( int i = 0; i < p.get_num_threads(); i++ ) {
               queue.push( k );
            }*/
            break;
        }
    }

    //Join threads
    for( int i = 0; i < tasks.size(); i++ )
       tasks[ i ].join();



}

template <typename GenFunc, typename TaskFunc>
inline void farm(parallel_execution_thr &p, GenFunc const &in, TaskFunc const & taskf ) {

    std::vector<std::thread> tasks;
//    Queue< typename std::result_of<GenFunc()>::type > queue(DEFAULT_SIZE);
    Queue< typename std::result_of<GenFunc()>::type > queue(DEFAULT_SIZE,p.is_lockfree());
    //Create threads
//    std::atomic<int> nend(0);
    for( int i = 0; i < p.get_num_threads(); i++ ) {
        tasks.push_back(
            std::thread(
                [&](){
                    // Register the thread in the execution model
                    p.register_thread();
                    typename std::result_of<GenFunc()>::type item;
                    item = queue.pop( );
                    while( item ) {
                       taskf( item.value() );
                       item = queue.pop( );
                    }
                    queue.push(item);
                    // Deregister the thread in the execution model
                    p.deregister_thread();
                }
            )
        );
    }

    //Generate elements
    while( 1 ) {
        auto k = in();
        queue.push( k ) ;
        if( !k ) {
/*            for( int i = 0; i < p.get_num_threads(); i++ ) {
                queue.push( k ) ;
            }*/
            break;
        }
    }

    //Join threads
    for( int i = 0; i < p.get_num_threads(); i++ )
       tasks[ i ].join();
}


template <typename TaskFunc>
FarmObj<parallel_execution_thr,TaskFunc> farm(parallel_execution_thr &p, TaskFunc && taskf){
   
   return FarmObj<parallel_execution_thr, TaskFunc>(p, taskf);
}
}
#endif
