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
    
    auto pop()
    {
//      while(consuming_.test_and_set());
      auto item = output_queues_[not_finished_[current_]].pop();
      if(!item.first) 
      {
        auto position = std::find(not_finished_.begin(), not_finished_.end(), not_finished_[current_]);
        if (position != not_finished_.end()){
          not_finished_.erase(position);
        }
        if(not_finished_.size() == 0) {
  //        consuming.clear();
          return T{item.first,-1};
        }
      }
      current_= (current_+1)%not_finished_.size();
   //   consuming.clear();
      return T{item.first,order++};
    }
    

    void push(typename T::first_type item, std::size_t queue_id) {
      output_queues_[queue_id].push(T{item,0});
    }


    joiner_queue(int num_producers, int q_size, queue_mode q_mode) :
      num_producers_{num_producers}/*, mut_{}, cond_var_{}*/
    {
      for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i);
      for(auto i = 0; i < num_producers_; i++) output_queues_.emplace_back(q_size,q_mode);
    }
 
    joiner_queue(joiner_queue && q) :
      output_queues_{q.output_queues_}, num_producers_{q.num_producers_}, /*mut_{}, cond_var_{},*/ not_finished_{q.not_finished_}
    { for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i);
    }
    
    joiner_queue(const joiner_queue &) = delete;
    joiner_queue & operator=(const joiner_queue &) = delete;

    std::vector<mpmc_queue<T>> output_queues_;

  private:
    int current_ = 0;
    long order = 0;
    int num_producers_ = 0;
    int end_ = 0;
    std::atomic_flag consuming_ = ATOMIC_FLAG_INIT;
 //   std::mutex mut_;
 //   std::condition_variable cond_var_;
    std::vector<int> not_finished_;
};

}
#endif
