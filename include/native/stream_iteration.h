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

#ifndef GRPPI_NATIVE_STREAM_ITERATION_H
#define GRPPI_NATIVE_STREAM_ITERATION_H

#include <thread>
#include <utility>
#include <memory>

#include "parallel_execution_native.h"

namespace grppi{ 

template<typename GenFunc, typename Predicate, typename OutFunc, typename ...Stages>
void stream_iteration(parallel_execution_native &p, GenFunc && in, pipeline_info<parallel_execution_native , Stages...> && se, Predicate && condition, OutFunc && out){
   mpmc_queue< typename std::result_of<GenFunc()>::type > queue(p.queue_size,p.lockfree);
   mpmc_queue< typename std::result_of<GenFunc()>::type > queueOut(p.queue_size,p.lockfree);
   std::atomic<int> nend (0);
   std::atomic<int> nelem (0);
   std::atomic<bool> sendFinish( false );
   //Stream generator
   std::thread gen([&](){
      // Register the thread in the execution model
      p.register_thread();
      while(1){
         auto k = in();
         if(!k){
             sendFinish=true;
             break;
         }
         nelem++;
         queue.push(k);
      }
      p.deregister_thread();
   });

   std::vector<std::thread> pipeThreads;
   composed_pipeline< mpmc_queue< typename std::result_of<GenFunc()>::type >, mpmc_queue< typename std::result_of<GenFunc()>::type >, 0, Stages ...>
      (queue, std::forward<pipeline_info<parallel_execution_native , Stages...> >(se) , queueOut, pipeThreads); 
 
   while(1){
      //If every element has been processed
      if(sendFinish&&nelem==0){
          queue.push(typename std::result_of<GenFunc()>::type{});
          sendFinish= false;
          break;
      }
      auto k = queueOut.pop();
      //Check the predicate
      if( !condition(k.value() ) ) {
           nelem--;
           out( k.value() );
       //If the condition is not met reintroduce the element in the input queue
      }else queue.push( k );

   }
   auto first = pipeThreads.begin();
   auto end = pipeThreads.end();
   gen.join();
   for(;first!=end;first++){ (*first).join();}

}

template<typename GenFunc, typename Operation, typename Predicate, typename OutFunc>
 void stream_iteration(parallel_execution_native &p, GenFunc && in, farm_info<parallel_execution_native,Operation> && se, Predicate && condition, OutFunc && out){
   std::vector<std::thread> tasks;
   mpmc_queue< typename std::result_of<GenFunc()>::type > queue(p.queue_size,p.lockfree);
   mpmc_queue< typename std::result_of<GenFunc()>::type > queueOut(p.queue_size,p.lockfree);
   std::atomic<int> nend (0);
   //Stream generator
   std::thread gen([&](){
      // Register the thread in the execution model
      se.exectype.register_thread();

      while(1){
         auto k = in();
         queue.push(k);
         if(!k) break;
      }
      //When generation is finished it starts working on the farm
      auto item = queue.pop();
      while(item){
         do{
             auto out = typename std::result_of<GenFunc()>::type( se.task(item.value()) );
             item = out;
         }while(condition(item.value()));
         queueOut.push(item);
         item = queue.pop();
      }
      nend++;
      if(nend == se.exectype.num_threads)
         queueOut.push( typename std::result_of<GenFunc()>::type ( ) );
      else queue.push(item);

      // Deregister the thread in the execution model
      se.exectype.deregister_thread();

   });
   //Farm workers
   for(int th = 1; th < se.exectype.num_threads; th++) {
     tasks.emplace_back([&]() {
       // Register the thread in the execution model
       se.exectype.register_thread();

       auto item = queue.pop();
       while (item) {
         do {
           auto out = typename std::result_of<GenFunc()>::type(se.task(item.value()));
           item = out;
         }
         while (condition(item.value()));
         queueOut.push(item);
         item = queue.pop();
       }
       nend++;
       if (nend == se.exectype.num_threads)
         queueOut.push(typename std::result_of<GenFunc()>::type());
       else 
         queue.push(item);

       // Deregister the thread in the execution model
       se.exectype.deregister_thread();
     });
   }
   //Output function
   std::thread outth([&](){
      while(1){
         auto k = queueOut.pop();
         if(!k) break;
         out(k.value());
      }
   });
   //Join threads
   auto first = tasks.begin();
   auto end = tasks.end();
   for(;first!=end;first++) (*first).join();
   gen.join();
   outth.join();

}

template<typename Generator, typename Operation, typename Predicate, typename Consumer>
void stream_iteration(parallel_execution_native &ex, Generator && gen, Operation && f, Predicate && condition, Consumer && cons){
  using namespace std;
  using generated_type = typename result_of<Generator()>::type;
  using generated_value_type = typename generated_type::value_type;

  using produced_type = typename result_of<Operation(generated_value_type)>::type;
  
  mpmc_queue<generated_type> generated_queue{ex.queue_size,ex.lockfree};
  mpmc_queue<std::experimental::optional<produced_type>> produced_queue{ex.queue_size,ex.lockfree};


  std::thread prod([&gen, &generated_queue](){
    while(1){
      auto k = gen();
      generated_queue.push(k);
      if(!k) break;
    }
  });
  
  std::thread task([&generated_queue,&f,&condition,&produced_queue](){
    auto item = generated_queue.pop();
    while(item){
     produced_type val = *item;
     do{
       val = f(val);
     }while(condition(val));
     produced_queue.push(std::experimental::optional<produced_type>{val});
     item = generated_queue.pop();
    }
    produced_queue.push(std::experimental::optional<produced_type>{});
  });

  std::thread sink([&cons,&produced_queue](){
    auto item = produced_queue.pop();
    while(item){
      cons(*item);
      item=produced_queue.pop();
    }
  });  
  prod.join();
  task.join();
  sink.join();

}

}
#endif
