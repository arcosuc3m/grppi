#ifndef GRPPI_OMP_SPLITTER_QUEUE_H
#define GRPPI_OMP_SPLITTER_QUEUE_H

#include "omp_mpmc_queue.h"

#include <omp.h>
#include <vector>
#include <atomic>
#include <condition_variable>

namespace grppi {

template <typename T, typename Queue, typename Splitting_policy>
class omp_splitter_queue{
  public:
    using value_type = T; 
    using queue_type = Queue;

    auto pop(int queue_id) {
      while (consumer_queues_[queue_id].is_empty()) 
      {
        //Only one consumer can take elements from the input queue
        std::unique_lock<std::mutex> lk(mut_);//omp_set_lock(&mut_);
        
        if(can_consume_ && consumer_queues_[queue_id].is_empty()) {
          can_consume_ = false;
          lk.unlock();
          //omp_unset_lock(&mut_);
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
          //omp_set_lock(&mut_);
          lk.lock();
          can_consume_ = true;
          lk.unlock();
          //omp_unset_lock(&mut_); 
        } 
        else {
          if(!can_consume_  && consumer_queues_[queue_id].is_empty()) {
             //omp_unset_lock(&mut_); 
             lk.unlock();
               #pragma omp taskyield
             //omp_set_lock(&mut_);
             lk.lock();
          }
          lk.unlock();
          //omp_unset_lock(&mut_); 
          
        }
        #pragma omp taskyield
      }
      auto pop_item = consumer_queues_[queue_id].pop();
      return std::move(pop_item);
      //return std::move(consumer_queues_[queue_id].pop());
    }
    
    void push(T item,int queue_id) {
      consumer_queues_[queue_id].push(item);
    }

    omp_splitter_queue(Queue & q, int num_queues, Splitting_policy policy, int q_size, queue_mode q_mode) :
      input_queue_{q}, policy_{policy}, num_consumers_{num_queues}
    {
      policy_.set_num_queues(num_consumers_);
      for (auto i = 0; i<num_queues; i++) {
        order_.push_back(0); 
        consumer_queues_.emplace_back(q_size,q_mode);
      }
      //omp_init_lock(&mut_);
    
    } 
    
    omp_splitter_queue(omp_splitter_queue && q) :
      input_queue_{q.input_queue_}, policy_{q.policy_}, num_consumers_{q.num_consumers_},order_{q.order_}
    { 
      policy_.set_num_queues(num_consumers_);
      for (auto i = 0; i<num_consumers_; i++) {
        consumer_queues_.emplace_back(std::move(q.consumer_queues_[i]));
      }
      //omp_init_lock(&mut_);
    }
    
    omp_splitter_queue(const omp_splitter_queue &) = delete;
    omp_splitter_queue & operator=(const omp_splitter_queue &) = delete;


  private:
    Queue& input_queue_;
    int num_consumers_;
    std::vector<mpmc_queue<T>> consumer_queues_;
    Splitting_policy policy_;
    //omp_lock_t mut_;
    std::mutex mut_;
    std::atomic_flag consuming_ = ATOMIC_FLAG_INIT;
    std::atomic<bool> can_consume_ {true};
    std::vector<long> order_;
};
}
#endif
