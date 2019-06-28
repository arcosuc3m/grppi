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


#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace grppi{

/**
\defgroup communication Communication
\brief Communication support types.
@{
*/

/**
\brief A lock-free multiple producer multiple consumer queue.
\tparam T Element type for the queue.
*/
template <typename T>
class atomic_mpmc_queue {
public:

  /// Type alias for element type.
  using value_type = T;

  /**
  \brief Constructs an atomic queue with a given size.
  \param size Size of the queue.
  */
  atomic_mpmc_queue(int size) :
    size_{size},
    buffer_{std::make_unique<T[]>(size)}
  {}

  /**
  \brief Move constructs an atomic queue from another one.
  \param q The queue to move from.
  */
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
  
  /**
  \brief Checks if the queue is empty.
  \return true if the queue is empty, false otherwise.
  */
  bool empty () const noexcept { 
    return pread_.load() == pwrite_.load();
  }

  /**
  \brief Pops an item from the queue.
  \return The value that has been extracted from the queue.
  \note This call may block by busy waiting if the queue is empty.
  */
  T pop () noexcept(std::is_nothrow_move_constructible<T>::value);

  /**
  \brief Pushes an element in the queue by move.
  \param item Value to be moved into the queue.
  \note This call may block by busy waiting if the queue is full.
  */
  void push (T && item) noexcept(std::is_nothrow_move_assignable<T>::value);

  /**
  \brief Pushes an element in the queue by copy.
  \param item Value to be copied into the queue.
  \note This call may block by busy waiting if the queue is full.
  */
  void push (T const & item) noexcept(std::is_nothrow_copy_assignable<T>::value);

private:
  /// Maximum number of elements in the queue.
  int size_;

  /// Buffer of elements.
  std::unique_ptr<T[]> buffer_;
  //std::vector<T> buffer_;

  /// Index to next position to read.
  std::atomic<unsigned long long> pread_{0};

  /// Index to next position to write.
  std::atomic<unsigned long long> pwrite_{0};

  /// Internal index to next position to read.
  std::atomic<unsigned long long> internal_pread_{0};

  /// Internal index to next position to write.
  std::atomic<unsigned long long> internal_pwrite_{0};
};

template <typename T>
T atomic_mpmc_queue<T>::pop() noexcept(std::is_nothrow_move_constructible<T>::value) 
{
  unsigned long long current;
  do {
    current = internal_pread_.load();
  } 
  while (!internal_pread_.compare_exchange_weak(current, current+1));
          
  while (current >= pwrite_.load()) {}

  auto item = std::move(buffer_[current%size_]); 
  auto aux = current;
  do {
    current = aux;
  } 
  while (!pread_.compare_exchange_weak(current, current+1));
     
  return item;
}

template <typename T>
void atomic_mpmc_queue<T>::push(T && item) noexcept(std::is_nothrow_move_assignable<T>::value) {
  unsigned long long current;
  do {
    current = internal_pwrite_.load();
  } 
  while (!internal_pwrite_.compare_exchange_weak(current, current+1));

  while (current >= (pread_.load()+size_)) {}

  buffer_[current%size_] = std::move(item);
  
  auto aux = current;
  do {
    current = aux;
  } 
  while (!pwrite_.compare_exchange_weak(current, current+1));
}

template <typename T>
void atomic_mpmc_queue<T>::push(T const & item) noexcept(std::is_nothrow_copy_assignable<T>::value) {
  unsigned long long current;
  do {
    current = internal_pwrite_.load();
  } 
  while (!internal_pwrite_.compare_exchange_weak(current, current+1));

  while (current >= (pread_.load()+size_)) {}

  buffer_[current%size_] = item;
  
  auto aux = current;
  do {
    current = aux;
  } 
  while (!pwrite_.compare_exchange_weak(current, current+1));
}

/**
\brief A lock-based multiple producer multiple consumer queue.
\tparam T Element type for the queue.
*/
template <typename T>
class locked_mpmc_queue {
public:

  /// Type alias for element type.
  using value_type = T;

  /**
  \brief Constructs a locked queue with a given size.
  \param size Size of the queue.
  */
  locked_mpmc_queue(int size) :
    size_{size},
    buffer_{std::make_unique<T[]>(size)},
    pread_{0},
    pwrite_{0}
  {}

  /**
  \brief Move constructs a locked queue from another one.
  \param q The queue to move from.
  */
  locked_mpmc_queue(locked_mpmc_queue && q) noexcept :
    size_{q.size_},
    buffer_{std::move(q.buffer_)},
    pread_{q.pread_},
    pwrite_{q.pwrite_}
  {}

  locked_mpmc_queue & operator=(locked_mpmc_queue && q) noexcept = delete;

  locked_mpmc_queue(locked_mpmc_queue const & q) noexcept = delete;
  locked_mpmc_queue & operator=(locked_mpmc_queue const & q) noexcept = delete;

   int size (){
    return size_;
   }
  
  /**
  \brief Checks if the queue is empty.
  \return true if the queue is empty, false otherwise.
  */
  bool empty () const noexcept { 
    return pread_== pwrite_;
  }

  /**
  \brief Pops an item from the queue.
  \return The value that has been extracted from the queue.
  \note This call may block through a mutex if the queue is empty.
  */
  T pop () noexcept(std::is_nothrow_move_constructible<T>::value);

  /**
  \brief Try to pop an item from the queue.
  \return If has been extracted from the queue.
  \note This call do not block the execution.
  */
  bool try_pop(T& item);
  /**
  \brief Pushes an element in the queue by move.
  \param item Value to be moved into the queue.
  \note This call may block through a mutex if the queue is empty.
  */
  void push (T && item) noexcept(std::is_nothrow_move_assignable<T>::value);

  /**
  \brief Pushes an element in the queue by copy.
  \param item Value to be copied into the queue.
  \note This call may block through a mutex if the queue is empty.
  */
  void push (T const & item) noexcept(std::is_nothrow_copy_assignable<T>::value);

private:
  /// Maximum number of elements in the queue.
  int size_;

  /// Buffer of elements.
  std::unique_ptr<T[]> buffer_;

  /// Index to next position to read.
  int pread_;

  /// Index to next position to write.
  int pwrite_;

  /// Mutex to synchronize access to the queue.
  std::mutex mut_{};

  /// Condition variable to signal empty queue.
  std::condition_variable empty_{};

  /// Condition variable to signal full queue.
  std::condition_variable full_{};
};

template <typename T>
bool locked_mpmc_queue<T>::try_pop(T& item) 
{
  std::unique_lock<std::mutex> lk(mut_);
  auto empty =  pread_ < pwrite_;
  if(empty){
    item = buffer_[pread_%size_];
    pread_++;
    full_.notify_one();
  }
  lk.unlock();
  return empty;
}

template <typename T>
T locked_mpmc_queue<T>::pop() noexcept(std::is_nothrow_move_constructible<T>::value) 
{
  std::unique_lock<std::mutex> lk(mut_);
//  std::cout<<"[POP before] pread " <<pread_ << " pwrite " <<pwrite_<<std::endl;
  
  empty_.wait(lk, [this] {return pread_ < pwrite_; });
//  std::cout<<"[POP after] pread " <<pread_ << " pwrite " <<pwrite_<<std::endl;
  auto item = std::move(buffer_[pread_%size_]);
  pread_++;
  lk.unlock();
  full_.notify_one();
  return item;
}

template <typename T>
void locked_mpmc_queue<T>::push(T && item) noexcept(std::is_nothrow_move_assignable<T>::value) 
{
  {
    std::unique_lock<std::mutex> lk(mut_);
//    std::cout<<"[PUSH] pread " <<pread_ << " pwrite " <<pwrite_<<std::endl;
    full_.wait(lk, [this] { return pwrite_ < (pread_ + size_); });
    buffer_[pwrite_%size_] = std::move(item);
    pwrite_++;
  }
  empty_.notify_one();
}

template <typename T>
void locked_mpmc_queue<T>::push(T const & item) noexcept(std::is_nothrow_copy_assignable<T>::value) 
{
  {
    std::unique_lock<std::mutex> lk(mut_);
    full_.wait(lk, [this] { return pwrite_ < (pread_ + size_); });
 //   std::cout<<"[PUSH] pread " <<pread_ << " pwrite " <<pwrite_<<std::endl;
    buffer_[pwrite_%size_] = item;
    pwrite_++;
  }
  empty_.notify_one();
}

/**
\brief Synchronization mode for queues.
*/
enum class queue_mode {
  /// Lock-free synchronization using atomics.
  lockfree, 
  /// Mutex based synchronization.
  blocking 
};

/**
\brief A multiple producer multiple consumer queue.
\tparam T Element type for the queue.
The mpmc_queue may be constructed providing a synchronization mode 
(lockfree or blocking).
*/
template <typename T>
class mpmc_queue{
public:
  using value_type = T;

  /**
  \brief Constructs a queue with a given size and a synchronization mode.
  \param size Size of the queue.
  \param mode Synchronization mode.
  */
  mpmc_queue(int size, queue_mode mode); 

  /**
  \brief Move constructs a queue from another one.
  \param q The queue to move from.
  */
  mpmc_queue(mpmc_queue && q); 

  mpmc_queue & operator=(mpmc_queue &&) = delete;

  mpmc_queue(const mpmc_queue &) = delete; 
  mpmc_queue & operator=(const mpmc_queue &) = delete;
    
  /**
  \brief Checks if the queue is empty.
  \return true if the queue is empty, false otherwise.
  */
  bool empty () const noexcept {
    return pself_const()->empty();
  }

  /**
  \brief Pops an item from the queue.
  \return The value that has been extracted from the queue.
  \note This call may block if the queue is empty.
  */
  T pop () noexcept(std::is_nothrow_move_constructible<T>::value) {
    return pself()->pop();
  }

  /**
  \brief Pushes an element in the queue by move.
  \param item Value to be moved into the queue.
  \note This call may block if the queue is empty.
  */
  void push (T && item) noexcept(std::is_nothrow_move_assignable<T>::value) {
    pself()->push(std::forward<T>(item));
  }

  /**
  \brief Pushes an element in the queue by copy.
  \param item Value to be copied into the queue.
  \note This call may block if the queue is empty.
  */
  void push (T const & item) noexcept(std::is_nothrow_copy_assignable<T>::value) {
    pself()->push(item);
  }

private:

  /**
  \brief Interface for polymorphic queue.
  */
  struct base_queue {
    virtual ~base_queue() noexcept = default;
    virtual bool empty() const noexcept = 0;
    virtual T pop () noexcept(std::is_nothrow_move_constructible<T>::value) = 0;
    virtual void push (T && item) noexcept(std::is_nothrow_move_assignable<T>::value) = 0;
    virtual void push (T const & item) noexcept(std::is_nothrow_copy_assignable<T>::value) = 0;
  };

  /**
  \brief Derived polymorphic interface to a concrete queue.
  \tparam Q Concrete queue to be wrapped.
  */
  template <typename Q>
  class concrete_queue : public base_queue {
  public:
    concrete_queue(int size) : queue_{size} {}
    concrete_queue(const concrete_queue<Q>&) = delete;
    concrete_queue(concrete_queue<Q>&&) = default;
    ~concrete_queue() = default;
    bool empty() const noexcept override { return queue_.empty(); }
    T pop () noexcept(std::is_nothrow_move_constructible<T>::value) override
      { return queue_.pop(); }
    void push (T && x) noexcept(std::is_nothrow_move_assignable<T>::value) override
      { queue_.push(std::forward<T>(x)); }
    void push (T const & x) noexcept(std::is_nothrow_copy_assignable<T>::value) override
      { queue_.push(x); }
  private:
    Q queue_;
  };

  /**
  \brief Get buffer containing queue wrapper as a pointer to base_queue.
  */
  base_queue * pself() noexcept {
    return reinterpret_cast<base_queue*>(&buffer_);
  }
      
  /**
  \brief Get buffer containing queue wrapper as a pointer to constant base_queue.
  */
  base_queue const * pself_const() const noexcept {
    return reinterpret_cast<base_queue const*>(&buffer_);
  }

  /// Type for concrete atomic queue.
  using concrete_atomic_queue = concrete_queue<atomic_mpmc_queue<T>>;

  /// Type for concrete locked queue.
  using concrete_locked_queue = concrete_queue<locked_mpmc_queue<T>>;

  /// Buffer that can hold any queue.
  std::aligned_union_t<0,
      concrete_atomic_queue,
      concrete_locked_queue> buffer_;
};

template <typename T>
mpmc_queue<T>::mpmc_queue(int size, queue_mode mode)
{
  switch (mode) {
    case queue_mode::lockfree:
      new (&buffer_) concrete_atomic_queue(size);
      break;
    case queue_mode::blocking:
      new (&buffer_) concrete_atomic_queue(size);
      break;
  }
}

template <typename T>
mpmc_queue<T>::mpmc_queue(mpmc_queue && q) 
{
  if (auto * patomic = dynamic_cast<concrete_atomic_queue*>(q.pself())) {
    new (&buffer_) concrete_atomic_queue{std::move(*patomic)};
  }
  else if (auto * plocked = dynamic_cast<concrete_locked_queue*>(q.pself())) {
    new (&buffer_) concrete_locked_queue{std::move(*plocked)};
  }
}

/**
@}
*/

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
