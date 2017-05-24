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


#include <thread>
#include <utility>
#include <memory>



namespace grppi{ 

template<typename GenFunc, typename Predicate, typename OutFunc, typename ...Stages>
inline void StreamIteration(parallel_execution_thr &p, GenFunc const & in, PipelineObj<parallel_execution_thr , Stages...> const & se, Predicate const & condition, OutFunc const & out){
   Queue< typename std::result_of<GenFunc()>::type > queue(DEFAULT_SIZE,p.lockfree);
   Queue< typename std::result_of<GenFunc()>::type > queueOut(DEFAULT_SIZE,p.lockfree);
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
   composedPipeline< Queue< typename std::result_of<GenFunc()>::type >, Queue< typename std::result_of<GenFunc()>::type >, 0, Stages ...>
      (queue, se, queueOut, pipeThreads); 
 
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

template<typename GenFunc, typename TaskFunc, typename Predicate, typename OutFunc>
inline void StreamIteration(parallel_execution_thr &p, GenFunc const & in, FarmObj<parallel_execution_thr,TaskFunc> const & se, Predicate const & condition, OutFunc const & out){
   std::vector<std::thread> tasks;
   Queue< typename std::result_of<GenFunc()>::type > queue(DEFAULT_SIZE,p.lockfree);
   Queue< typename std::result_of<GenFunc()>::type > queueOut(DEFAULT_SIZE,p.lockfree);
   std::atomic<int> nend (0);
   //Stream generator
   std::thread gen([&](){
      // Register the thread in the execution model
      se.exectype->register_thread();

      while(1){
         auto k = in();
         queue.push(k);
         if(!k) break;
      }
      //When generation is finished it starts working on the farm
      auto item = queue.pop();
      while(item){
         do{
             auto out = typename std::result_of<GenFunc()>::type( (*se.task)(item.value()) );
             item = out;
         }while(condition(item.value()));
         queueOut.push(item);
         item = queue.pop();
      }
      nend++;
      if(nend == se.exectype->num_threads)
         queueOut.push( typename std::result_of<GenFunc()>::type ( ) );
      else queue.push(item);

      // Deregister the thread in the execution model
      se.exectype->deregister_thread();

   });
   //Farm workers
   for(int th = 1; th < se.exectype->num_threads; th++) {
      tasks.push_back(
          std::thread([&](){
              // Register the thread in the execution model
              se.exectype->register_thread();

              auto item = queue.pop();
              while(item){
                  do{
                      auto out = typename std::result_of<GenFunc()>::type ( (*se.task)(item.value()) );
                      item = out;
                  }while(condition(item.value()));
                  queueOut.push(item);
                  item = queue.pop();
              }
              nend++;
              if(nend == se.exectype->num_threads)
                  queueOut.push( typename std::result_of<GenFunc()>::type ( ) );
              else queue.push(item);

              // Deregister the thread in the execution model
              se.exectype->deregister_thread();
          }
      ));
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

}
/*template<typename GenFunc, typename TaskFunc, typename Predicate, typename OutFunc>
inline void StreamIteration(sequential_execution, GenFunc const & in, TaskFunc const & f, Predicate const & condition, OutFunc const & out){
   while(1){
       auto k = in();
       if(!k) break;
       while(condition(k)){
          k = f(k);
       }
       out(k);
   }
}
*/
