#ifndef GRPPI_COMMON_JOINER_QUEUE_H
#define GRPPI_COMMON_JOINER_QUEUE_H

#include "mpmc_queue.h"

#include <vector>
#include <atomic>
#include <condition_variable>

namespace grppi {

template <typename T>
class joiner_queue{
  public:
    using value_type = T; 
    
    T pop()
    {
      while(added_queues_ != num_producers_);
      //while(consuming_.test_and_set());
      //auto item = std::remove_reference<mpmc_queue<T>&>::type(output_queues_[not_finished_[current_]]).pop();
      auto item = output_queues_[not_finished_[current_]].get().pop();
      while(!item.first) 
      {
        auto position = std::find(not_finished_.begin(), not_finished_.end(), not_finished_[current_]);
        if (position != not_finished_.end()){
          not_finished_.erase(position);
        }
        if(not_finished_.size() == 0) {
       // consuming_.clear();
          std::unique_lock<std::mutex> lock(mut_);
          finished=true;
          cond_var_.notify_all();
          return T{item.first,-1};
        }
        current_= (current_+1)%not_finished_.size();
        auto item = output_queues_[not_finished_[current_]].get().pop();
      }
      current_= (current_+1)%not_finished_.size();
     // consuming_.clear();
      return T{item.first,order++};
    }

    void wait(){
      std::unique_lock<std::mutex> lock(mut_);
      while (!finished) {  // loop to avoid spurious wakeups
        cond_var_.wait(lock);
      }               
    }
   
    void push(typename T::first_type item, std::size_t queue_id) {
      output_queues_[queue_id].get().push(T{item,0});
    }

    void add_queue(mpmc_queue<T> & q, int index){
      output_queues_[index] = q;
      added_queues_++;
    }

    joiner_queue(int num_producers, int q_size, queue_mode q_mode) :
      num_producers_{num_producers}/*, mut_{}, cond_var_{}*/
    {
      for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i);
      //std::vector<mpmc_queue<T>&> output_queues_ = std::vector<mpmc_queue<T>&>(num_producers_);
      //for(auto i = 0; i < num_producers_; i++) output_queues_.emplace_back(q_size,q_mode);
      output_queues_.reserve(num_producers_);
    }
 
    joiner_queue(joiner_queue && q) :
      output_queues_{q.output_queues_},/* num_producers_{q.num_producers_},*/ /*mut_{}, cond_var_{},*/ not_finished_{q.not_finished_}
    { 
      for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i);
    }
    
    joiner_queue(const joiner_queue &) = delete;
    joiner_queue & operator=(const joiner_queue &) = delete;

    std::vector<std::reference_wrapper<mpmc_queue<T>>> output_queues_;

  private:
    int current_ = 0;
    long order = 0;
    int num_producers_ = 0;
    int end_ = 0;
    bool finished{false};
    std::atomic<int> added_queues_{0};
    std::atomic_flag consuming_ = ATOMIC_FLAG_INIT;
    std::mutex mut_;
    std::condition_variable cond_var_;
    std::vector<int> not_finished_;
};

namespace internal
{

template <typename, template <typename ...> class>
struct is_join_queue : std::false_type{};

template <class... T, template <class...> class W>
struct is_join_queue <W<T...>, W> :std::true_type{};

}

template <typename T>
constexpr bool is_join_queue = internal::is_join_queue<std::decay_t<T>, joiner_queue>();

template <typename T>
using requires_join_queue = std::enable_if_t<is_join_queue<T>,int>;

template <typename T>
using requires_no_join_queue = std::enable_if_t<!is_join_queue<T>,int>;
}
#endif
