#ifndef GRPPI_OMP_JOINER_QUEUE_H
#define GRPPI_OMP_JOINER_QUEUE_H

#include "omp_mpmc_queue.h"

#include <omp.h>
#include <vector>
#include <atomic>
#include <condition_variable>

namespace grppi {

template <typename T>
class omp_joiner_queue{
  public:
    using value_type = T; 

    auto pop() {
      return output_queue_.pop();
    }
    

    void push(typename T::first_type item, std::size_t queue_id) {
      while(!omp_test_lock(&lk)) {
         #pragma omp taskyield
      }
      while( queue_id != not_finished_[current_] && item ){
        omp_unset_lock(&lk);
        #pragma omp taskyield
        while(!omp_test_lock(&lk)) {
          #pragma omp taskyield
        }
         
      }
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
      omp_unset_lock(&lk);
    }


    omp_joiner_queue(int num_producers, int q_size, queue_mode q_mode) :
      num_producers_{num_producers}, output_queue_{q_size,q_mode}
    { for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i);
      omp_init_lock(&lk);
    }
 
    omp_joiner_queue(omp_joiner_queue && q) :
      output_queue_{q.output_queue_}, num_producers_{q.num_producers_},  not_finished_{q.not_finished_}
    { for( auto i = 0; i < num_producers_; i++) not_finished_.push_back(i);
      omp_init_lock(&lk);
    }
    
    omp_joiner_queue(const omp_joiner_queue &) = delete;
    omp_joiner_queue & operator=(const omp_joiner_queue &) = delete;

    omp_mpmc_queue<T> output_queue_;

  private:
    int current_ = 0;
    long order = 0;
    int num_producers_ = 0;
    int end_ = 0;
    omp_lock_t lk;
    std::vector<int> not_finished_;
};

}
#endif
