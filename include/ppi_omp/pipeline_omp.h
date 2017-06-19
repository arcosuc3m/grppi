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

#ifndef GRPPI_PIPELINE_OMP_H
#define GRPPI_PIPELINE_OMP_H

#ifdef GRPPI_OMP

#include <boost/lockfree/spsc_queue.hpp>

using namespace std;
namespace grppi{
//Last stage
template <typename Stream, typename Stage>
void stages( parallel_execution_omp p, Stream& st, Stage && s ){
    //Start task
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
   //End task
}

template <typename Task, typename Stream,typename... Stages>
 void stages( parallel_execution_omp p, Stream& st, FilterObj<parallel_execution_omp, Task>& se, Stages ... sgs ) {
    if(p.ordering){
       Queue< typename Stream::value_type > q(DEFAULT_SIZE);

       std::atomic<int> nend ( 0 );
       for( int th = 0; th < se.exectype->num_threads; th++){
           #pragma omp task shared(q,se,st,nend)
           {
                 typename Stream::value_type item;
                 item = st.pop( ) ;
                 while( item.first ) {
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
           }
       }
       Queue< typename Stream::value_type > qOut(DEFAULT_SIZE);
       #pragma omp task shared (qOut,q)
       {
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
       }
       stages(p, qOut, sgs ... );
       #pragma omp taskwait
      }else{
       Queue< typename Stream::value_type > q(DEFAULT_SIZE);

       std::atomic<int> nend ( 0 );
       for( int th = 0; th < se.exectype->num_threads; th++){
             #pragma omp task shared(q,se,st,nend)
             {
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

          }
       }
       stages(p, q, sgs ... );
       #pragma omp taskwait
    }




}

template <typename Task, typename Stream,typename... Stages>
 void stages( parallel_execution_omp p, Stream& st, FarmObj<parallel_execution_omp, Task> se, Stages ... sgs ) {
   
    Queue< std::pair < optional < typename std::result_of< Task(typename Stream::value_type::first_type::value_type) >::type >, long > > q(DEFAULT_SIZE);
    std::atomic<int> nend ( 0 );
    for( int th = 0; th < se.exectype->num_threads; th++){
      #pragma omp task shared(nend,q,se,st)
      {
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
      }              
    }
    stages(p, q, sgs ... );
    #pragma omp taskwait
}




//Intermediate stages
template <typename Stage, typename Stream,typename ... Stages>
void stages(parallel_execution_omp p, Stream& st, Stage && se, Stages ... sgs ) {

    //Create new queue
//    boost::lockfree::spsc_queue< optional< typename std::result_of< Stage(typename Stream::value_type::value_type) > ::type>, boost::lockfree::capacity<BOOST_QUEUE_SIZE>> q;
    Queue<std::pair< optional <typename std::result_of<Stage(typename Stream::value_type::first_type::value_type)>::type >, long >> q(DEFAULT_SIZE);
    //Start task
    #pragma omp task shared( se, st, q )
    {
        typename Stream::value_type item;
        item = st.pop( ); 
        while( item.first ) {
            auto out = optional <typename std::result_of<Stage(typename Stream::value_type::first_type::value_type)>::type > ( se(item.first.value()) );

            q.push( std::make_pair(out, item.second) );
            item = st.pop(  ) ;
        }
        q.push( std::make_pair(optional< typename std::result_of< Stage(typename Stream::value_type::first_type::value_type) > ::type>(),-1) ) ;
    }
    //End task
    //Create next stage
    stages(p, q, sgs ... );
//    #pragma omp taskwait
}

//First stage
template <typename FuncIn, typename = typename std::result_of<FuncIn()>::type,
          typename ...Stages,
          requires_no_arguments<FuncIn> = 0>
void pipeline(parallel_execution_omp p, FuncIn && in, Stages ... sts ) {

    //Create first queue
    Queue<std::pair< typename std::result_of<FuncIn()>::type, long>> q(DEFAULT_SIZE);

    //Create stream generator stage
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task shared(in,q)
            {
                long order = 0;
                while( 1 ) {
                    auto k = in();
                    q.push( std::make_pair(k,order) ) ;
                    order++;
                    if( !k ) 
                        break;
                }
            }
            //Create next stage
            stages(p, q, sts ... );
            #pragma omp taskwait
        }
    }
}

}
#endif

#endif