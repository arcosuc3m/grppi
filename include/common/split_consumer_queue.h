#ifndef GRPPI_COMMON_SPLIT_CONSUMER_QUEUE_H
#define GRPPI_COMMON_SPLIT_CONSUMER_QUEUE_H

namespace grppi{

template <typename Queue_type>
class split_consumer_queue{
  public:
     using value_type = typename std::decay<Queue_type>::type::value_type;
     auto pop(){
       return std::move(splitter_queue_.pop(queue_id_));
     }
     
     void push(auto &&item) {
       splitter_queue_.push( std::move(item), queue_id_ );
     }
     
     split_consumer_queue(Queue_type& queue, std::size_t id) :
       splitter_queue_{queue}, queue_id_{id} { } 
  private:
     std::size_t queue_id_;
     Queue_type& splitter_queue_;

};

}

#endif
