#ifndef GRPPI_OMP_WINDOWER_QUEUE_H
#define GRPPI_OMP_WINDOWER_QUEUE_H

#include "omp_mpmc_queue.h"

#include <omp.h>

#include <vector>
#include <atomic>
#include <condition_variable>
#include <tuple>
#include <experimental/optional>

namespace grppi {

template <typename InQueue,typename Window>
class omp_windower_queue{
  public:
//    using value_type = std::vector<typename std::decay<InQueue>::type::value_type>;
    using window_type = typename std::result_of<decltype(&Window::get_window)(Window)>::type;
    using window_optional_type = std::experimental::optional<window_type>;
    using value_type = std::pair <window_optional_type, long> ;
    
    void push(){
       
    }

    auto pop() { 
      using namespace std;
      using namespace experimental;
      if(!end_){
        while(!omp_test_lock(&mut_)){
          #pragma omp taskyield 
        } 
        auto item = input_queue_.pop();
        if(item.first){
          while(!policy_.add_item(std::move(*item.first) )){
            item = input_queue_.pop();
            if(!item.first){
               end_ = true;
               break;
            }
          }
          auto window = make_pair(make_optional(policy_.get_window()), order_);
          order_++;
          omp_unset_lock(&mut_);
          return window;
        }
        end_ = true;
        omp_unset_lock(&mut_);
      }
      return make_pair(window_optional_type{}, -1);
   }
    
    omp_windower_queue(InQueue & q, Window w) :
      input_queue_{q}, policy_{w} {
      omp_init_lock(&mut_);
    }
 
    omp_windower_queue(omp_windower_queue && q) :
      input_queue_{q.input_queue_}, policy_{policy_} 
    {
      omp_init_lock(&mut_);
    }  
    
    omp_windower_queue(const omp_windower_queue &) = delete;
    omp_windower_queue & operator=(const omp_windower_queue &) = delete;

  private:
    omp_lock_t mut_;
    Window policy_;
    InQueue& input_queue_;    
    int order_ = 0;
    bool end_ = false;
};

}
#endif
