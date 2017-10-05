#ifndef GRPPI_COMMON_WINDOWER_QUEUE_H
#define GRPPI_COMMON_WINDOWER_QUEUE_H

#include "mpmc_queue.h"

#include <vector>
#include <atomic>
#include <condition_variable>
#include <tuple>
#include <experimental/optional>

namespace grppi {

template <typename InQueue,typename Window>
class windower_queue{
  public:
//    using value_type = std::vector<typename std::decay<InQueue>::type::value_type>;
    using window_type = typename std::result_of<decltype(&Window::get_window)(Window)>::type;
    using window_optional_type = std::experimental::optional<window_type>;
    using value_type = std::pair <window_optional_type, long> ;
    
    void push(value_type w){
       
    }

    auto pop() { 
      using namespace std;
      using namespace experimental;
     // mut_.lock();
      while(consuming_.test_and_set());
      if(!end_){
        auto item = input_queue_.pop();
        if(item.first){
          while(!policy_.add_item(std::move(*item.first) )){
            item = input_queue_.pop();
            if(!item.first){
               end_ = true;
               consuming_.clear();
  //    mut_.unlock();
               return std::move(make_pair(window_optional_type{}, -1));
            }
          }
          auto window = make_pair(make_optional(policy_.get_window()), order_);
          order_++;
          consuming_.clear();
  //    mut_.unlock();
          return std::move(window);
        }
        end_ = true;
      }
  //    mut_.unlock();
      consuming_.clear();
      return std::move(make_pair(window_optional_type{}, -1));
   }
    
    windower_queue(InQueue & q, Window w) :
      input_queue_{q}, policy_{w} {}
 
    windower_queue(windower_queue && q) :
      input_queue_{q.input_queue_}, policy_{policy_} {}
      
    
    windower_queue(const windower_queue &) = delete;
    windower_queue & operator=(const windower_queue &) = delete;

  private:
std::atomic_flag consuming_ = ATOMIC_FLAG_INIT;
    std::mutex mut_;
    Window policy_;
    InQueue& input_queue_;    
    int order_ = 0;
    bool end_ = false;
};

}
#endif
