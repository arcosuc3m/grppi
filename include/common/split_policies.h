#ifndef GRPPI_COMMON_SPLIT_POLICIES_H
#define GRPPI_COMMON_SPLIT_POLICIES_H

#include <vector>

namespace grppi {

class duplicate{
  public:
    inline void set_num_queues(int n){
      num_queues_ = n;
      for(int i = 0; i < num_queues_; i++) next.push_back(i);
    }
    
    inline auto& next_queue() {
      //std::vector<int> next;
      return next;
    }
  private:
    std::vector<int> next;
    int num_queues_ = 0;
};


class round_robin{
  public:
    void set_num_queues(int n){
      num_queues_ = n;
    }

    auto& next_queue() {

      next[0] = current_q_;
      current_item_ += 1;
      if (current_item_ == items_per_queue_){
        current_item_ = 0;
        current_q_ = (current_q_+1) % num_queues_;
      } 
      return next;
    }
   
    round_robin(int num_items) : items_per_queue_ { num_items } {
      next.push_back(current_q_);
    }
  private:
    int num_queues_ = 0;
    int current_q_ = 0;
    int items_per_queue_ = 0;
    int current_item_ = 0;
    std::vector<int> next;

};


}
#endif

