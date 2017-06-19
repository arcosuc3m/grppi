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

#ifndef GRPPI_STREAM_FILTER_THR_H
#define GRPPI_STREAM_FILTER_THR_H

using namespace std;
namespace grppi{
template <typename GenFunc, typename FilterFunc, typename OutFunc>
 void stream_filter(parallel_execution_thr &p, GenFunc && in, FilterFunc && filter, OutFunc && out ) {

    std::vector<std::thread> tasks;

    Queue< std::pair< typename std::result_of<GenFunc()>::type, long> > queue(DEFAULT_SIZE,p.lockfree);
    Queue< std::pair< typename std::result_of<GenFunc()>::type, long> > outqueue(DEFAULT_SIZE,p.lockfree);

    //THREAD 1-(N-1) EXECUTE FILTER AND PUSH THE VALUE IF TRUE
    for(int i=0; i< p.num_threads - 1; i++){
      tasks.push_back(
          std::thread(
            [&](){

               // Register the thread in the execution model
               p.register_thread();

               std::pair< typename std::result_of<GenFunc()>::type,long > item;
               //dequeue a pair element - order
               item = queue.pop();
               while( item.first ){
                   if(filter(item.first.value()))
                       //If is an acepted element
                       outqueue.push(item);
                   else{
                       //If is a discarded element
                       outqueue.push(std::make_pair(typename std::result_of<GenFunc()>::type(),item.second));
                   }
                   item = queue.pop();
               }
               //If is the last element
               outqueue.push(std::make_pair(item.first,0));

               // Deregister the thread in the execution model
               p.deregister_thread();
            }
         )
       );
     }

//LAST THREAD CALL FUNCTION OUT WITH THE FILTERED ELEMENTS
    tasks.push_back(
      std::thread(
        [&](){

           // Register the thread in the execution model
           p.register_thread();

           int nend = 0;
           std::pair<typename std::result_of<GenFunc()>::type, long> item;
           std::vector< std::pair<typename std::result_of<GenFunc()>::type, long> > aux_vector;
           long order = 0;
           //Dequeue an element
           item = outqueue.pop();
           while(nend != p.num_threads - 1){
              //If is an end of stream element
              if(!item.first&&item.second==0){
                  nend++;
                  if(nend == p.num_threads -1 ) break;
              }
              //If there is not an end element
              else {
                  //If the element is the next one to be procesed
                  if(order == item.second){
                      if(item.first)
                         out(item.first.value());
                      order++;
                  }else{
                      //If the incoming element is disordered
                      aux_vector.push_back(item);
                  }
              }
              //Search in the vector for next elements
              for(auto it = aux_vector.begin(); it < aux_vector.end();it++) {
                  if((*it).second == order){
                       if((*it).first)
                          out((*it).first.value());
                       aux_vector.erase(it);
                       order++;
                  }
              } 
              item = outqueue.pop();
           }
           while(aux_vector.size()>0){
               for(auto it = aux_vector.begin(); it < aux_vector.end();it++) {
                  if((*it).second == order){
                       if((*it).first)
                          out((*it).first.value());
                       aux_vector.erase(it);
                       order++;
                  }
              }
           }
           
           // Deregister the thread in the execution model
           p.deregister_thread();
        }
      )
    );

    //THREAD 0 ENQUEUE ELEMENTS
    long order = 0;
    while(1){
        auto k = in();
        queue.push(make_pair(k,order));
        order++;
        if( k.end ){
           for(int i = 0; i< p.num_threads -1; i++){
              queue.push(make_pair(k,0));
           }
           break;
        }
    }

    for(int i = 0; i< p.num_threads; i++) tasks[i].join();

}

template <typename FilterFunc>
FilterObj<parallel_execution_thr, FilterFunc> stream_filter(parallel_execution_thr &p, FilterFunc && taskf){
   return FilterObj<parallel_execution_thr, FilterFunc>(p, taskf);

}
}
#endif
