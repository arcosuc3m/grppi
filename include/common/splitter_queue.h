#ifndef GRPPI_COMMON_SPLITTER_QUEUE_H
#define GRPPI_COMMON_SPLITTER_QUEUE_H

#include "mpmc_queue.h"

#include <vector>
#include <atomic>
#include <condition_variable>

#include <thread>

namespace grppi {

template <typename T, typename Queue, typename Splitting_policy>
class splitter_queue{
  public:
    using value_type = T; 
    using queue_type = Queue;


    auto pop(int queue_id) {
      while (consumer_queues_[queue_id].is_empty()) 
      {
        //Only one consumer can take elements from the input queue
       // std::unique_lock<std::mutex> lock(mut_);
        while(consuming_.test_and_set());
        if(can_consume_ && consumer_queues_[queue_id].is_empty()) {
          can_consume_ = false;
         // lock.unlock();
          consuming_.clear();
          do {
            auto next = policy_.next_queue();
            auto item = input_queue_.pop();

            //If is an end of stream item
            if(!item.first) {
              for(auto i = 0;i<num_consumers_; i++){ 
                consumer_queues_[i].push(item);
              }
            } 
            else {
              for( auto it = next.begin(); it != next.end(); it++){
                item.second = order_[*it];
                order_[*it]++;
                consumer_queues_[*it].push(item);
              }
            }

            //Wake up all the conusmer threads to check if they have any item to conusme
          } while(consumer_queues_[queue_id].is_empty());

          //This is used to avoid potential everlasting waiting threads
          can_consume_ = true;
        } 
        else {
          consuming_.clear();
          while(!can_consume_ && consumer_queues_[queue_id].is_empty()) {
          }
        }
      }
      auto pop_item = consumer_queues_[queue_id].pop();
      return std::move(pop_item);
      //return std::move(consumer_queues_[queue_id].pop());
    }
    
    void push(T && item,int queue_id) {
      consumer_queues_[queue_id].push(item);
    }

    splitter_queue(Queue & q, int num_queues, Splitting_policy policy, int q_size, queue_mode q_mode) :
      input_queue_{q}, policy_{policy}, num_consumers_{num_queues}
    {
      for (auto i = 0; i<num_queues; i++) {
        order_.push_back(0); 
        consumer_queues_.emplace_back(q_size,q_mode);
      }

    } 
    
    splitter_queue(splitter_queue && q) :
      input_queue_{q.input_queue_}, policy_{q.policy_}, num_consumers_{q.num_consumers_},order_{q.order_}
    { 
      policy_.set_num_queues(num_consumers_);
      for (auto i = 0; i<num_consumers_; i++) {
        consumer_queues_.emplace_back(std::move(q.consumer_queues_[i]));
      }
    }
    
    splitter_queue(const splitter_queue &) = delete;
    splitter_queue & operator=(const splitter_queue &) = delete;

  
  private:
    Queue& input_queue_;
    std::vector<mpmc_queue<T>> consumer_queues_;
    int num_consumers_;
    Splitting_policy policy_;
    std::vector<long> order_;
    std::atomic_flag consuming_ = ATOMIC_FLAG_INIT;
    std::atomic<bool> can_consume_ {true};
};
}
#endif
