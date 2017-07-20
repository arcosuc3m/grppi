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

#ifndef GRPPI_NATIVE_FARM_H
#define GRPPI_NATIVE_FARM_H

#include <experimental/optional>

#include <thread>
#include <utility>
#include <memory>

#include "parallel_execution_native.h"

namespace grppi{

template <typename Generator, typename Consumer, typename ...Stages>
void farm(parallel_execution_native &p, Generator && gen, pipeline_info<parallel_execution_native,Stages ...> && pipe, Consumer && cons)
{
  std::vector<std::thread> tasks;
  using input_type = typename std::result_of<Generator()>::type;
  using input_value_type = typename input_type::value_type;
  using input_queue_type = mpmc_queue< input_type >;
  input_queue_type in_queue(p.queue_size,p.lockfree);

  using output_type = typename get_return_type<input_value_type,Stages...>::type;
  using output_queue_type = mpmc_queue< std::experimental::optional<output_type>>;
  output_queue_type out_queue (p.queue_size,p.lockfree);

  std::vector<std::thread> pipe_threads;
  for( int i = 0; i < p.num_threads; i++ ) {
    composed_pipeline<input_queue_type,output_queue_type,0,Stages...>
      (in_queue,std::forward<pipeline_info<parallel_execution_native,Stages...>>(pipe),out_queue,pipe_threads);
  }
  
  //Consumer function
  std::thread sink(
    [&](){
      // Register the thread in the execution model
      p.register_thread();
      int finished_threads = 0;
      do 
      {
        auto item = out_queue.pop();
        if( item) cons(*item);
        else finished_threads++;
      } while( finished_threads != p.num_threads );
      // Deregister the thread in the execution model
      p.deregister_thread();
    }
  );  

  //Generator function
  for(;;) {
    auto k = gen();
    in_queue.push( k );
    if( !k ) {
      for( int i = 1; i < p.num_threads; i++ ) {
        in_queue.push( k );
      }
      break;
    }
  }
  
  for(auto && t : pipe_threads) t.join();
  sink.join();





}



template <typename Generator, typename Operation, typename Consumer>
void farm(parallel_execution_native &p, Generator &&gen, Operation && op , Consumer &&cons) {

    std::vector<std::thread> tasks;
    mpmc_queue< typename std::result_of<Generator()>::type > queue (p.queue_size,p.lockfree);
    mpmc_queue< std::experimental::optional < typename std::result_of<Operation(typename std::result_of<Generator()>::type::value_type)>::type > > queueout(p.queue_size, p.lockfree);
    std::atomic<int> nend(0);
    //Create threads
    for (int i = 0; i < p.num_threads; i++) {
      tasks.emplace_back([&]() {
        // Register the thread in the execution model
        p.register_thread();

        typename std::result_of<Generator()>::type item;
        item = queue.pop() ;
        while (item) {
          auto out = op( item.value() );
          queueout.push( std::experimental::optional < typename std::result_of<Operation(typename std::result_of<Generator()>::type::value_type)>::type >(out) );
          item = queue.pop() ;
        }
        queue.push(item);

        nend++;
        if (nend == p.num_threads)
           queueout.push( std::experimental::optional< typename std::result_of<Operation(typename std::result_of<Generator()>::type::value_type)>::type >() ) ;

        // Deregister the thread in the execution model
        p.deregister_thread();
      });
    }

    //SINK 
    tasks.emplace_back([&]() {
      // Register the thread in the execution model
      p.register_thread();

      std::experimental::optional< typename std::result_of<Operation(typename std::result_of<Generator()>::type::value_type)>::type > item;
      item = queueout.pop() ;
      while (item) {
        cons(item.value());
        item = queueout.pop();
      }

      // Deregister the thread in the execution model
      p.deregister_thread();
    });

   //Generate elements
    while( 1 ) {
        auto k = gen();
        queue.push( k );
        if( !k ) {
/*            for( int i = 0; i < p.num_threads; i++ ) {
               queue.push( k );
            }*/
            break;
        }
    }

    //Join threads
    for( int i = 0; i < tasks.size(); i++ )
       tasks[ i ].join();



}

template <typename Generator, typename Operation>
 void farm(parallel_execution_native &p, Generator &&gen, Operation && op ) {

    std::vector<std::thread> tasks;
    mpmc_queue< typename std::result_of<Generator()>::type > queue(p.queue_size,p.lockfree);
    //Create threads
//    std::atomic<int> nend(0);
    for( int i = 0; i < p.num_threads; i++ ) {
      tasks.emplace_back([&]() {
        // Register the thread in the execution model
        p.register_thread();

        typename std::result_of<Generator()>::type item;
        item = queue.pop();
        while (item) {
          op(item.value());
          item = queue.pop();
        }
        queue.push(item);

        // Deregister the thread in the execution model
        p.deregister_thread();
      });
    }

    //Generate elements
    while( 1 ) {
        auto k = gen();
        queue.push( k ) ;
        if( !k ) {
/*            for( int i = 0; i < p.num_threads; i++ ) {
                queue.push( k ) ;
            }*/
            break;
        }
    }

    //Join threads
    for( int i = 0; i < p.num_threads; i++ )
       tasks[ i ].join();
}


template <typename Operation>
farm_info<parallel_execution_native,Operation> farm(parallel_execution_native &p, Operation && op){
   
   return farm_info<parallel_execution_native, Operation>(p, std::forward<Operation>(op) );
}

}
#endif
