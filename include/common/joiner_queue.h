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

    auto pop() {
      return output_queue_.pop();
    }
    

    void push(typename T::first_type item, std::size_t queue_id) {
      std::unique_lock<std::mutex> lock{mut_};
      while( queue_id != not_finished_[current_] && item ) cond_var_.wait(lock);
      if(!item) {
         auto position = std::find(not_finished_.begin(), not_finished_.end(), queue_id);
         if (position != not_finished_.end()){
           not_finished_.erase(position);
         }
      }
      if(item) {
        output_queue_.push( T{item,order} );
        order++; 
        current_= (current_+1)%not_finished_.size();
      }
      else {
        if(not_finished_.size() != 0) current_= (current_+1)%not_finished_.size();
        else output_queue_.push( T{item,-1} );
      }
      cond_var_.notify_all();
    }


    joiner_queue(int num_producers, int q_size, queue_mode q_mode) :
      num_producers_{num_producers}, output_queue_{q_size,q_mode}, mut_{}, cond_var_{}
    { for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i); }
 
    joiner_queue(joiner_queue && q) :
      output_queue_{q.output_queue_}, num_producers_{q.num_producers_}, mut_{}, cond_var_{}, not_finished_{q.not_finished_}
    { for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i); }
    
    joiner_queue(const joiner_queue &) = delete;
    joiner_queue & operator=(const joiner_queue &) = delete;

    mpmc_queue<T> output_queue_;

  private:
    int current_ = 0;
    long order = 0;
    int num_producers_ = 0;
    int end_ = 0;
    std::mutex mut_;
    std::condition_variable cond_var_;
    std::vector<int> not_finished_;
};

}
#endif
