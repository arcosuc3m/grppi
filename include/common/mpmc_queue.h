/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GRPPI_COMMON_MPMC_QUEUE_H
#define GRPPI_COMMON_MPMC_QUEUE_H


#include <vector>
#include <atomic>
#include <iostream>
#include <mutex>
#include <condition_variable>

namespace grppi{



enum class queue_mode {lockfree = true, blocking = false};

template <typename T>
class mpmc_queue{

   public:
      using value_type = T;

      mpmc_queue<T>(int q_size, queue_mode q_mode ) :
          size{q_size}, 
          buffer{std::vector<T>(q_size)}, 
          mode{q_mode}, 
          pread{0}, 
          pwrite{0}, 
          internal_pread{0}, 
          internal_pwrite{0} 
      {}
      
      mpmc_queue(mpmc_queue && q) :
        size{q.size},
        buffer{std::move(q.buffer)},
        mode{q.mode},
        pread{q.pread.load()},
        pwrite{q.pwrite.load()},
        internal_pread{q.internal_pread.load()},
        internal_pwrite{q.internal_pwrite.load()},
        m{},
        empty{},
        full{}
      {}

      mpmc_queue(const mpmc_queue &) = delete; 
      mpmc_queue & operator=(const mpmc_queue &) = delete;
    
      bool is_empty () const noexcept;
      T pop () ;
      bool push (T item) ;

   private:
      bool is_full (unsigned long long current) const noexcept;
      bool is_empty (unsigned long long current) const noexcept;

      int size;
      std::vector<T> buffer;
      queue_mode mode;

      std::atomic<unsigned long long> pread;
      std::atomic<unsigned long long> pwrite;
      std::atomic<unsigned long long> internal_pread;
      std::atomic<unsigned long long> internal_pwrite;


      std::mutex m {};
      std::condition_variable empty{};
      std::condition_variable full{};

};


template <typename T>
bool mpmc_queue<T>::is_empty() const noexcept {
    return pread.load()==pwrite.load();
}

template <typename T>
T mpmc_queue<T>::pop(){
  if(mode == queue_mode::lockfree){
    
     unsigned long long current;

     do{
        current = internal_pread.load();
     }while(!internal_pread.compare_exchange_weak(current, current+1));
          
     while(is_empty(current));

     auto item = std::move(buffer[current%size]); 
     auto aux = current;
     do{
        current = aux;
     }while(!pread.compare_exchange_weak(current, current+1));
     
     return std::move(item);
  }else{
     
     std::unique_lock<std::mutex> lk(m);
     while(is_empty(pread)){
        empty.wait(lk);
     }  
     auto item = std::move(buffer[pread%size]);
     pread++;    
     lk.unlock();
     full.notify_one();
     
     return std::move(item);
  }

}

template <typename T>
bool mpmc_queue<T>::push(T item){
  if(mode == queue_mode::lockfree){
     unsigned long long current;
     do{
         current = internal_pwrite.load();
     }while(!internal_pwrite.compare_exchange_weak(current, current+1));

     while(is_full(current));

     buffer[current%size] = std::move(item);
  
     auto aux = current;
     do{
        current = aux;
     }while(!pwrite.compare_exchange_weak(current, current+1));

     return true;
  }else{

    std::unique_lock<std::mutex> lk(m);
    while(is_full(pwrite)){
        full.wait(lk);
    }
    buffer[pwrite%size] = std::move(item);

    pwrite++;
    lk.unlock();
    empty.notify_one();

    return true;
  }
}

template <typename T>
bool mpmc_queue<T>::is_empty(unsigned long long current) const noexcept {
  if(current >= pwrite.load()) return true;
  return false;
}

template <typename T>
bool mpmc_queue<T>::is_full(unsigned long long current) const noexcept{
  if(current >= (pread.load()+size)) return true;
  return false;

}

namespace internal {

template <typename T>
struct is_queue : std::false_type {};

template <typename T>
struct is_queue<mpmc_queue<T>> : std::true_type {};

}

template <typename T>
constexpr bool is_queue = internal::is_queue<T>();

template <typename T>
using requires_queue = std::enable_if_t<is_queue<T>, int>;

}

#endif
