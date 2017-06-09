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

#ifndef GRPPI_PIPELINE_THR_H
#define GRPPI_PIPELINE_THR_H

#include <thread>

using namespace std;
namespace grppi{

template <typename InStream, typename OutStream, int currentStage, typename ...Stages>
inline typename std::enable_if<(currentStage == (sizeof...(Stages)-1)), void>::type composed_pipeline(InStream& qin, PipelineObj<parallel_execution_thr, Stages...> const & pipe, OutStream &qout,std::vector<std::thread> & tasks)
{
      composed_pipeline((*pipe.exectype), qin, std::get<currentStage>(pipe.stages), qout, tasks);
}


template <typename InStream, typename OutStream, int currentStage, typename ...Stages>
inline typename std::enable_if<(currentStage < (sizeof...(Stages)-1)), void>::type composed_pipeline(InStream& qin, PipelineObj<parallel_execution_thr, Stages...> const & pipe, OutStream & qout,std::vector<std::thread> & tasks)
{
      typedef typename std::tuple_element<currentStage, decltype(pipe.stages)>::type lambdaPointerType;
      typedef typename std::remove_reference<decltype(*lambdaPointerType())>::type  lambdaType; 
      typedef typename std::result_of< lambdaType (typename InStream::value_type::value_type) > ::type queueType;

      static Queue<optional<queueType>> queueOut(DEFAULT_SIZE,(pipe.exectype)->lockfree); 

      composed_pipeline((*pipe.exectype), qin, std::get<currentStage>(pipe.stages), queueOut, tasks);
      composed_pipeline<Queue<optional<queueType>>,OutStream, currentStage+1, Stages ...>(queueOut,pipe,qout,tasks);
       
}

template <typename InStream, typename Stage, typename OutStream>
inline void composed_pipeline(parallel_execution_thr &p, InStream &qin, Stage const & s, OutStream &qout, std::vector<std::thread> & tasks){
    tasks.push_back(
      std::thread([&](){
        typedef typename std::remove_reference<decltype(*Stage())>::type  lambdaType;
        p.register_thread();
        
        auto item = qin.pop();
        while(true){
           if(!item) {
                qout.push( optional< typename std::result_of< lambdaType(typename InStream::value_type::value_type) >::type >() );
                break;
           }else{
//              if(!item) qin.push(item); 
//              else{
                auto out = optional< typename std::result_of< lambdaType(typename InStream::value_type::value_type) >::type >( (*s)(item.value()) );
                //   std::cout<<" pipe thread pushing" << out.value()<<std::endl;
                qout.push(out);
//              }
           }
           item = qin.pop();
        }

        p.deregister_thread();
      }
    ));
}

//Last stage
template <typename Stream, typename Stage>
inline void stages( parallel_execution_thr & p,Stream& st, Stage const& s ) {

   p.register_thread();

   typename Stream::value_type item;
   std::vector<typename Stream::value_type> elements;
   long current = 0;
   if(p.ordering){
     item = st.pop( );
     while( item.first ) {
        if(current == item.second){
           s( item.first.value() );
           current ++;
        }else{
           elements.push_back(item);
        }
        for(auto it = elements.begin(); it != elements.end(); it++){
           if((*it).second == current) {
              s((*it).first.value());
              elements.erase(it);
              current++;
              break;
           }
        }
       item = st.pop( );
     }
     while(elements.size()>0){
        for(auto it = elements.begin(); it != elements.end(); it++){
          if((*it).second == current) {
            s((*it).first.value());
            elements.erase(it);
            current++;
            break;
          }
        }
      }
   }else{
      item = st.pop( );
      while( item.first ) {
        s( item.first.value() );
        item = st.pop( );
     }
   }
  
   p.deregister_thread();

}

//Stream reduce stage
template <typename Task, typename Red, typename Stream>
inline void stages( parallel_execution_thr &p, Stream& st, ReduceObj<parallel_execution_thr, Task, Red>& se) {
    std::vector<std::thread> tasks;
    Queue<typename std::result_of< Task(typename Stream::value_type) >::type > queueOut(DEFAULT_SIZE,p.lockfree);

    for( int th = 0; th < se.exectype.num_threads; th++){
        tasks.push_back(
           std::thread([&](){
              typename Stream::value_type item;
              item = st.pop( );
              while( item ) {
                 auto local =  (*se.task)(item) ;
                 queueOut.push( local ) ;
                 item = st.pop( );
              }
              typename std::result_of< Task(typename Stream::value_type) >::type out;
              queueOut.push( out ) ;
       }));
    }
    //stages(p, q, sgs ... );
    for(int i=0;i<tasks.size(); i++) tasks[i].join();
}

//Filtering stage
template <typename Task, typename Stream, typename... Stages>
inline void stages( parallel_execution_thr &p, Stream& st, FilterObj<parallel_execution_thr, Task> se, Stages ... sgs ) {
    
    std::vector<std::thread> tasks;
    if(p.ordering){
       Queue< typename Stream::value_type > q(DEFAULT_SIZE,p.lockfree);

       std::atomic<int> nend ( 0 );
       for( int th = 0; th < se.exectype->num_threads; th++){
          tasks.push_back(
              std::thread([&](){
                 //Register the thread in the execution model
                 se.exectype->register_thread();
                 typename Stream::value_type item;
                 item = st.pop( ) ;
                 while( item.first ) {
                  //MODIFIED from *se->task
                     if( (*se.task)(item.first.value()) )
                        q.push( item );
                     else{
                        q.push( std::make_pair( typename Stream::value_type::first_type()  ,item.second) );
                     } 
                     item = st.pop();
                 }
                 nend++;
                 if(nend == se.exectype->num_threads){
                    q.push( std::make_pair(typename Stream::value_type::first_type(), -1) );
                 }else{
                    st.push(item);
                 }
                 //MODIFIED from se->exectype.deregister_thread();
                 //Deregister the thread in the execution model
                 se.exectype->deregister_thread();

          }));
       } 
       Queue< typename Stream::value_type > qOut(DEFAULT_SIZE,p.lockfree);
       auto orderingthr = std::thread([&](){
          p.register_thread();
          typename Stream::value_type item;
          std::vector<typename Stream::value_type> elements;
          int current = 0;
          long order = 0;
          item = q.pop( ) ;
          while(1){
             if(!item.first && item.second == -1){ 
                 break; 
             }
             if(item.second == current){
                if(item.first){
                   qOut.push(std::make_pair(item.first,order));
                   order++;
                }
                current++;
             }else{
                elements.push_back(item);
             }
             for(auto it = elements.begin(); it < elements.end(); it++){
                if((*it).second == current){
                    if((*it).first){
                        qOut.push(std::make_pair((*it).first,order));
                        order++;
                    }
                    elements.erase(it);
                    current++;
                    break;
                }
             }
             item=q.pop();
          }
          while(elements.size()>0){
            for(auto it = elements.begin(); it < elements.end(); it++){
              if((*it).second == current){
                  if((*it).first){ 
                     qOut.push(std::make_pair((*it).first,order));
                     order++;
                  }
                  elements.erase(it);
                  current++;
                  break;
              }
            }
          }
          qOut.push(item);
   p.deregister_thread();
       });
       stages(p, qOut, sgs ... );
       orderingthr.join();
    }else{
       Queue< typename Stream::value_type > q(DEFAULT_SIZE, p.lockfree);

       std::atomic<int> nend ( 0 );
       for( int th = 0; th < se.exectype->num_threads; th++){
          tasks.push_back(
              std::thread([&](){
                  //Register the thread in the execution model
                  se.exectype->register_thread();
                 typename Stream::value_type item;
                 item = st.pop( ) ;
                 while( item.first ) {
                     if( (*se.task)(item.first.value()) ) 
                        q.push( item );
//                     else{
//                        q.push( std::make_pair( typename Stream::value_type::first_type()  ,item.second) );
//                     } 
                      item = st.pop();
                 }
                 nend++;
                 if(nend == se.exectype->num_threads){
                    q.push( std::make_pair(typename Stream::value_type::first_type(), -1) );
                 }else{
                    st.push(item);
                 }
                 //Deregister the thread in the execution model
                 se.exectype->deregister_thread();
               
          }));
       }
       stages(p, q, sgs ... );
    }
    for(int i=0;i<tasks.size(); i++) tasks[i].join();


}

//Farm stage
template <typename Task, typename Stream, typename... Stages>
inline void stages( parallel_execution_thr &p, Stream& st, FarmObj<parallel_execution_thr, Task> se, Stages ... sgs ) {
    std::vector<std::thread> tasks;
    //Queue<std::pair< optional <typename std::result_of<Stage(typename Stream::value_type::value_type)>::type >, long > q(DEFAULT_SIZE);
    Queue< std::pair < optional < typename std::result_of< Task(typename Stream::value_type::first_type::value_type) >::type >, long > > q(DEFAULT_SIZE,p.lockfree);
    std::atomic<int> nend ( 0 );
    for( int th = 0; th < se.exectype->num_threads; th++){
          tasks.push_back(
              std::thread([&](){
                  //Register the thread in the execution model
                  se.exectype->register_thread();

                  long order = 0;
                  auto item = st.pop(); 
                  while( item.first ) {
                      auto out = optional< typename std::result_of< Task(typename Stream::value_type::first_type::value_type) >::type >( (*se.task)(item.first.value()) );
                      
                      q.push( std::make_pair(out,item.second)) ;
                      item = st.pop( ); 
                 }
                 st.push(item);
                 nend++;
                 if(nend == se.exectype->num_threads) 
                      q.push(make_pair(optional< typename std::result_of< Task(typename Stream::value_type::first_type::value_type) >::type >(), -1));
                
                 //Deregister the thread in the execution model
                 se.exectype->deregister_thread();
             })
          );
    }
    stages(p, q, sgs ... );
    
    for(int i=0;i<tasks.size(); i++) tasks[i].join();

}

//Intermediate stages
template <typename Stage, typename Stream,typename... Stages>
inline void stages( parallel_execution_thr &p,Stream& st, Stage const& se, Stages ... sgs ) {

    //Create new queue

    Queue<std::pair< optional <typename std::result_of<Stage(typename Stream::value_type::first_type::value_type)>::type >, long >> q(DEFAULT_SIZE,p.lockfree);

    //Start task
    std::thread task( 
        [&](){
            //Register the thread in the execution model
            p.register_thread();
            long order = 0;
            typename Stream::value_type item;
    //        while( !st.try_dequeue( item ) );
            item = st.pop( );
            while( item.first ) {
                auto out = optional <typename std::result_of<Stage(typename Stream::value_type::first_type::value_type)>::type > ( se(item.first.value()) );
                //auto out = se(item.value());
                q.push( std::make_pair(out, item.second)) ;
                item = st.pop( ) ;
            }
        	q.push( std::make_pair(optional< typename std::result_of< Stage(typename Stream::value_type::first_type::value_type) > ::type>(),-1) ) ;
            //Deregister the thread in the execution model
            p.deregister_thread();
        }
    );

    //End task
    //Create next stage
    stages(p, q, sgs ... );
    task.join();
}

//First stage
//template <typename FuncIn,typename... Arguments>
template <typename FuncIn, typename ...Stages,
          requires_no_arguments<FuncIn> = 0>
void pipeline( parallel_execution_thr& p, FuncIn const & in, Stages ... sts ) {
    //Create first queue
    Queue<std::pair< typename std::result_of<FuncIn()>::type, long>> q(DEFAULT_SIZE,p.lockfree);
    //Create stream generator stage
    std::thread task(
        [&](){
            //Register the thread in the execution model
            p.register_thread();
            long order = 0;
            while( 1 ) {
                auto k = in();
                q.push( std::make_pair(k , order) ) ;
                order++;
                if ( !k )
                    break;
            }
            //Deregister the thread in the execution model
            p.deregister_thread();
        }
    );

    //Create next stage
    stages(p, q, sts ... );
    task.join();
}



//template <typename Stage, typename = typename std::enable_if<!std::is_convertible<Stage,execution_model>::value>::type, typename ...Stages>
//PipelineObj<Execution_model,Stage,Stages...> pipeline(Execution_model &p, Stage && s, Stages && ...sts)
template <typename Execution_model, typename Stage,  
          /*typename std::enable_if<_has_arguments<Stage>::value>::type,*/  
          typename ...Stages,
          requires_arguments<Stage> = 0>
PipelineObj< Execution_model,Stage,Stages...> pipeline(Execution_model &p, Stage && s, Stages && ...sts)
{
    return PipelineObj<Execution_model,Stage, Stages ...> (p, s, sts...);
}
}
#endif
