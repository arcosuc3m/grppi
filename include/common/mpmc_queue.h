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

template <typename T>
class atomic_mpmc_queue {
public:

  using value_type = T;

  atomic_mpmc_queue(int size) :
    size_{size},
    buffer_{std::vector<T>(size)}
  {}

  atomic_mpmc_queue(atomic_mpmc_queue && q) noexcept :
    size_{q.size_},
    buffer_{std::move(q.buffer_)},
    pread_{q.pread_.load()},
    pwrite_{q.pwrite_.load()},
    internal_pread_{q.internal_pread_.load()},
    internal_pwrite_{q.internal_pwrite_.load()}
  {}

  atomic_mpmc_queue & operator=(atomic_mpmc_queue && q) noexcept = delete;

  atomic_mpmc_queue(atomic_mpmc_queue const & q) noexcept = delete;
  atomic_mpmc_queue & operator=(atomic_mpmc_queue const & q) noexcept = delete;
  
  bool empty () const noexcept { 
    return pread_.load() == pwrite_.load();
  }

  T pop () noexcept(std::is_nothrow_move_constructible<T>::value);
  void push (T item) noexcept(std::is_nothrow_move_constructible<T>::value);

private:
  int size_;
  std::vector<T> buffer_;

  std::atomic<unsigned long long> pread_{0};
  std::atomic<unsigned long long> pwrite_{0};
  std::atomic<unsigned long long> internal_pread_{0};
  std::atomic<unsigned long long> internal_pwrite_{0};
};

template <typename T>
T atomic_mpmc_queue<T>::pop() noexcept(std::is_nothrow_move_constructible<T>::value) {
  unsigned long long current;
  do {
    current = internal_pread_.load();
  } while(!internal_pread_.compare_exchange_weak(current, current+1));
          
  while(current >= pwrite_.load());

  auto item = std::move(buffer_[current%size_]); 
  auto aux = current;
  do {
    current = aux;
  } while(!pread_.compare_exchange_weak(current, current+1));
     
  return item;
}

template <typename T>
// TODO: What should be the best way of passing item
void atomic_mpmc_queue<T>::push(T item) noexcept(std::is_nothrow_move_constructible<T>::value) {
  unsigned long long current;
  do{
    current = internal_pwrite_.load();
  } while(!internal_pwrite_.compare_exchange_weak(current, current+1));

  while (current >= (pread_.load()+size_));

  buffer_[current%size_] = std::move(item);
  
  auto aux = current;
  do {
    current = aux;
  } while(!pwrite_.compare_exchange_weak(current, current+1));
}

template <typename T>
class locked_mpmc_queue {
public:

  using value_type = T;

  locked_mpmc_queue(int size) :
    size_{size},
    buffer_{std::vector<T>(size)},
    pread_{0},
    pwrite_{0}
  {}

  locked_mpmc_queue(locked_mpmc_queue && q) noexcept :
    size_{q.size_},
    buffer_{std::move(q.buffer_)},
    pread_{q.pread_.load()},
    pwrite_{q.pwrite_.load()}
  {}

  locked_mpmc_queue & operator=(locked_mpmc_queue && q) noexcept = delete;

  locked_mpmc_queue(locked_mpmc_queue const & q) noexcept = delete;
  locked_mpmc_queue & operator=(locked_mpmc_queue const & q) noexcept = delete;
  
  bool empty () const noexcept { 
    return pread_.load() == pwrite_.load();
  }

  T pop () noexcept(std::is_nothrow_move_constructible<T>::value);
  void push (T item) noexcept(std::is_nothrow_move_constructible<T>::value);

private:
  bool is_full (unsigned long long current) const noexcept;
  bool is_empty (unsigned long long current) const noexcept;

private:
  int size_;
  std::vector<T> buffer_;

  std::atomic<unsigned long long> pread_;
  std::atomic<unsigned long long> pwrite_;

  std::mutex mut_{};
  std::condition_variable empty_{};
  std::condition_variable full_{};
};

template <typename T>
T locked_mpmc_queue<T>::pop() noexcept(std::is_nothrow_move_constructible<T>::value) {
  std::unique_lock<std::mutex> lk(mut_);
  while(pread_.load() >= pwrite_.load()) {
    empty_.wait(lk);
  }  
  auto item = std::move(buffer_[pread_%size_]);
  pread_++;    
  lk.unlock();
  full_.notify_one();
     
  return item;
}

template <typename T>
// TODO: What should be the best way of passing item
void locked_mpmc_queue<T>::push(T item) noexcept(std::is_nothrow_move_constructible<T>::value) {
  std::unique_lock<std::mutex> lk(mut_);
  while (pwrite_.load() >= (pread_.load() + size_)) {
    full_.wait(lk);
  }
  buffer_[pwrite_%size_] = std::move(item);

  pwrite_++;
  lk.unlock();
  empty_.notify_one();
}

enum class queue_mode {lockfree = true, blocking = false};

template <typename T>
class mpmc_queue{

   public:
      using value_type = T;

      mpmc_queue<T>(int q_size, queue_mode q_mode ) :
        mode{q_mode}, 
        atomic_queue_{q_size},
        locked_queue_{q_size}
      {}
      
      mpmc_queue(mpmc_queue && q) :
        mode{q.mode},
        atomic_queue_{std::move(q.atomic_queue_)},
        locked_queue_{std::move(q.locked_queue_)}
      {}
      mpmc_queue & operator=(mpmc_queue &&) = delete;

      mpmc_queue(const mpmc_queue &) = delete; 
      mpmc_queue & operator=(const mpmc_queue &) = delete;
    
      bool is_empty () const noexcept;
      T pop ();
      bool push (T item) ;

   private:
      queue_mode mode;
      atomic_mpmc_queue<T> atomic_queue_;
      locked_mpmc_queue<T> locked_queue_;
};



template <typename T>
bool mpmc_queue<T>::is_empty() const noexcept {
  if (mode==queue_mode::lockfree) {
    return atomic_queue_.empty();
  }
  else {
    return locked_queue_.empty();
  }
}

template <typename T>
T mpmc_queue<T>::pop(){
  if(mode == queue_mode::lockfree){
    return atomic_queue_.pop();
  }
  else{
    return locked_queue_.pop();
  }
}

template <typename T>
bool mpmc_queue<T>::push(T item){
  if(mode == queue_mode::lockfree){
    atomic_queue_.push(item);
    return true;
  }
  else {
    locked_queue_.push(item);
    return true;
  }
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
