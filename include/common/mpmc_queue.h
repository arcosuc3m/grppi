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

#include <vector>
#include <atomic>
#include <iostream>
#include <mutex>
#include <condition_variable>

namespace grppi{

template <typename T>
class mpmc_queue{
   private:
      int size;
      std::vector<T> buffer;
      std::atomic<unsigned long long> pread;
      std::atomic<unsigned long long> pwrite;
      std::atomic<unsigned long long> internal_pwrite;
      std::atomic<unsigned long long> internal_pread;

      bool lockfree = false;

      std::mutex mutex;
      std::condition_variable emptycv;
      std::condition_variable fullcv;

      bool full(unsigned long long current);
      bool empty(unsigned long long current);
   public:  
      typedef T value_type;
      T pop();
      bool push(T item);
      bool empty();
      
      mpmc_queue<T>(int _size, bool active){
         size = _size;
         buffer = std::vector<T>(size);
         pread = 0;
         pwrite = 0;
         internal_pread = 0;
         internal_pwrite = 0;
         lockfree = active;
      }

      mpmc_queue<T>(int _size){
         size = _size;
         buffer = std::vector<T>(size);
         pread = 0;
         pwrite = 0;
         internal_pread = 0;
         internal_pwrite = 0;
      }
};


template <typename T>
bool mpmc_queue<T>::empty(){
    return pread.load()==pwrite.load();
}


template <typename T>
bool mpmc_queue<T>::empty(unsigned long long current){
  if(current >= pwrite.load()) return true;  
  return false;
}

template <typename T>
bool mpmc_queue<T>::full(unsigned long long current){
  if(current >= (pread.load()+size)) return true;
  return false;

}
template <typename T>
T mpmc_queue<T>::pop(){
  if(lockfree){
    
     unsigned long long current;

     do{
        current = internal_pread.load();
     }while(!internal_pread.compare_exchange_weak(current, current+1));
          
     while(empty(current));

     auto item = std::move(buffer[current%size]); 
     auto aux = current;
     do{
        current = aux;
     }while(!pread.compare_exchange_weak(current, current+1));
     
     return std::move(item);
  }else{
     
     std::unique_lock<std::mutex> lk(mutex);
     while(empty(pread)){
        emptycv.wait(lk);
     }  
     auto item = std::move(buffer[pread%size]);
     pread++;    
     lk.unlock();
     fullcv.notify_one();
     
     return std::move(item);
  }

}

template <typename T>
bool mpmc_queue<T>::push(T item){
  if(lockfree){

     unsigned long long current;
     do{
         current = internal_pwrite.load();
     }while(!internal_pwrite.compare_exchange_weak(current, current+1));

     while(full(current));

     buffer[current%size] = std::move(item);
  
     auto aux = current;
     do{
        current = aux;
     }while(!pwrite.compare_exchange_weak(current, current+1));

     return true;
  }else{

    std::unique_lock<std::mutex> lk(mutex);
    while(full(pwrite)){
        fullcv.wait(lk);
    }
    buffer[pwrite%size] = std::move(item);

    pwrite++;
    lk.unlock();
    emptycv.notify_one();

    return true;
  }

}

}
