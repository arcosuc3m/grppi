/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GRPPI_ZMQ_TASK_H
#define GRPPI_ZMQ_TASK_H

#include <set>
#include <iostream>
#include <sstream>
#include <string>

#include <zmq.hpp>

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
//#pragma GCC diagnostic pop

#include "zmq_data_reference.h"
#include "zmq_serialization.h"

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

/**
  \brief Distributed definition for a task.
  This type defines a task as the id of the function to be applied,
  an id for the task and its dependencies.

  This type is used as an example of a task type.
*/
class zmq_task{
  public:
    // Type alias for data reference.
    using data_ref_type = zmq_data_reference;

    /**
    \brief Construct an empty task.

    Creates a task with the end function and task ids.
    */
    zmq_task(): function_id_{-1}, task_id_{-1}, order_{-1}, local_ids_{}, is_hard_{false}, data_location_{}, task_serialized_{}, is_task_serialized_{false} {
    COUT << "TASK: (-1,-1,-1) (0,false,-1,-1) CREATED" << ENDL;
    };

    
    /**
    \brief Construct a task.

    Creates a task with the function and task ids received as parameters.

    \param f_id function id
    \param t_id task id
    \param order order of the data element of the pipeline
    \param local_ids list of node ids for local execution
    \param is_hard flag:can only execute on local nodes
    */
    zmq_task(long f_id, long t_id, long order,
             std::vector<long> local_ids, bool is_hard):
      function_id_{f_id}, task_id_{t_id}, order_{order}, local_ids_{local_ids}, is_hard_{is_hard}, data_location_{}, task_serialized_{}, is_task_serialized_{false} {
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << local_ids_.size() << "," <<  local_ids_.data() << "," << is_hard_ << "," <<  data_location_.size() << ") CREATED" << ENDL;

    }
    
    /**
    \brief Construct a task.

    Creates a task with the function and task ids received as parameters.

    \param f_id function id
    \param t_id task id
    \param order order of the data element of the pipeline
    \param local_ids list of node ids for local execution
    \param is_hard flag:can only execute on local nodes
    \param data_location data location reference
    */
    zmq_task(long f_id, long t_id, long order,
                 std::vector<long> local_ids, bool is_hard,
                 std::vector<data_ref_type> ref):
      function_id_{f_id}, task_id_{t_id}, order_{order}, local_ids_{local_ids}, is_hard_{is_hard}, data_location_{ref}, task_serialized_{}, is_task_serialized_{false} {
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << local_ids_.size() << "," <<  local_ids_.data() << "," << is_hard_ << "," <<  data_location_.size() << "," <<  data_location_[0].get_id() << "," << data_location_[0].get_pos() << ") CREATED" << ENDL;
      }
//#define __DEBUG__
#ifdef __DEBUG__
    zmq_task(const zmq_task & task) :
        function_id_{task.function_id_},
        task_id_{task.task_id_},
        order_{task.order_},
        local_ids_{task.local_ids_},
        is_hard_{task.is_hard_},
        data_location_{task.data_location_},
        before_dep_{task.before_dep_},
        after_dep_{task.after_dep_},
        is_task_serialized_{task.is_task_serialized_},
        task_serialized_{task.task_serialized_} {
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << task.local_ids_.data() << "," << local_ids_.data() << ") COPIED" << ENDL;
    }

    zmq_task(zmq_task && task) :
        function_id_{std::move(task.function_id_)},
        task_id_{std::move(task.task_id_)},
        order_{std::move(task.order_)},
        local_ids_{std::move(task.local_ids_)},
        is_hard_{std::move(task.is_hard_)},
        data_location_{std::move(task.data_location_)},
        before_dep_{std::move(task.before_dep_)},
        after_dep_{std::move(task.after_dep_)},
        is_task_serialized_{std::move(task.is_task_serialized_)},
        task_serialized_{std::move(task.task_serialized_)} {
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << task.local_ids_.data() << "," << local_ids_.data() << ") MOVED" << ENDL;
    }

    zmq_task& operator = (const zmq_task & task) {
        function_id_ = task.function_id_;
        task_id_ = task.task_id_;
        order_ = task.order_;
        local_ids_ = task.local_ids_;
        is_hard_ = task.is_hard_;
        data_location_ = task.data_location_;
        before_dep_ = task.before_dep_;
        after_dep_ = task.after_dep_;
        is_task_serialized_ = task.is_task_serialized_;
        task_serialized_ = task.task_serialized_;
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << task.local_ids_.data() << "," << local_ids_.data() << ") COPY ASSIG." << ENDL;
      return *this;
    }

    zmq_task& operator = (zmq_task && task) {
        function_id_ = std::move(task.function_id_);
        task_id_ = std::move(task.task_id_);
        order_ = std::move(task.order_);
        local_ids_ = std::move(task.local_ids_);
        is_hard_ = std::move(task.is_hard_);
        data_location_ = std::move(task.data_location_);
        before_dep_ = std::move(task.before_dep_);
        after_dep_ = std::move(task.after_dep_);
        is_task_serialized_ = std::move(task.is_task_serialized_);
        task_serialized_ = std::move(task.task_serialized_);
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << task.local_ids_.data() << "," << local_ids_.data() << ") MOVE ASSIG." << ENDL;
      return *this;
    }
#endif

    inline bool operator==(const zmq_task& rhs) const {
      COUT << "zmq_task::operator==" << ENDL;
      return task_id_ == rhs.task_id_;
    }

    inline bool operator!=(const zmq_task& rhs) const {
      COUT << "zmq_task::operator!=" << ENDL;
      return task_id_ != rhs.task_id_;
    } 

    inline bool operator>(const zmq_task& rhs) const {
      COUT << "zmq_task::operator>" << ENDL;
      return task_id_ > rhs.task_id_;
    }

    inline bool operator<(const zmq_task& rhs) const {
      COUT << "zmq_task::operator<" << ENDL;
      return task_id_ < rhs.task_id_;
    }
    
    /**
    \brief Return the id of the task function.
    
    \return  id of the task function
    */
    long get_id() const
    {
      return function_id_;
    }

    /**
    \brief Return the task id.

    \return  task id.
    */
    long get_task_id() const
    {
      return task_id_;
    }
    
    /**
    \brief Return the input data location of the task.
    \return  input data location of the task.
    */
    std::vector<data_ref_type> get_data_location() const
    {
      return data_location_;
    }

    /**
    \brief Store the input data location of the task.
    \param loc input data location of the task.
    */
    void set_data_location(std::vector<data_ref_type> loc)
    {
      data_location_ = loc;
      //reset serialized data
      is_task_serialized_= false;
    }

    /**
    \brief Return the order number of the task.

    \return  order number of the task
    */
    long get_order() const
    {
      return order_;
    }

    /**
    \brief Store the order number of the task.

    \param order order number of the task
    */
    void set_order(long order)
    {
      order_ = order;
      //reset serialized data
      is_task_serialized_= false;
    }

    /**
    \brief Return the list of node ids for local execution.

    \return list of node ids for local execution
    */
    std::vector<long> get_local_ids() const
    {
      return local_ids_;
    }

    /**
    \brief Store the list of node ids for local execution.

    \param local_ids list of node ids for local execution
    */
    void set_local_ids(std::vector<long> local_ids)
    {
      local_ids_ = local_ids;
      //reset serialized data
      is_task_serialized_= false;
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << local_ids_.size() << "," <<  local_ids_.data() << ") LOCAL_IDS SET" << ENDL;

    }

    /**
    \brief Return the flag:can only execute on local nodes.

    \return flag:can only execute on local nodes
    */
    bool get_is_hard() const
    {
      return is_hard_;
    }

    /**
    \brief Store the flag:can only execute on local nodes.

    \param is_hard flag:can only execute on local nodes
    */
    void set_is_hard(bool is_hard)
    {
      is_hard_ = is_hard;
      //reset serialized data
      is_task_serialized_= false;
    }

    /**
    \brief Return the before dependencies of the task.
    \return before dependencies of the task.
    */
    std::set<long> get_before_dep() const
    {
      return before_dep_;
    }

    /**
    \brief Store the before dependencies of the task.
    \param dep before dependencies of the task.
    */
    void set_before_dep(std::set<long> dep)
    {
      before_dep_ = dep;
      //reset serialized data
      is_task_serialized_= false;
    }

    /**
    \brief Return the after dependencies of the task.
    \return after dependencies of the task.
    */
    std::set<long> get_after_dep() const
    {
      return after_dep_;
    }

    /**
    \brief Store the after dependencies of the task.
    \param dep after dependencies of the task.
    */
    void set_after_dep(std::set<long> dep)
    {
      after_dep_ = dep;
      //reset serialized data
      is_task_serialized_= false;
    }

    /**
    \brief Construct a task from the serialize vector.

    Construct a task from the serialize vector.
    \param data serialize  data
    \param size serialize  size
    */
    void set_serialized(char * data, long size)
    {
       (*this) = internal::deserialize<zmq_task>(data,size);
    }
    
    /**
    \brief Get the serialize vector for the task.
    
    Get the serialize vector for the task.
    \return task serialize vector
    */
    std::vector<char> get_serialized() const
    {
      return internal::serialize(*this);
    }

    /**
    \brief Send a task over a ZMQ socket.

    Send a task over a ZMQ socket.
    \param socket zmq socket
    \param flags sending flags (ZMQ_SNDMORE,ZMQ_NOBLOCK)
    */
    void send(zmq::socket_t &socket, int flags = 0) const
    {
      // if not done, serialize task
      if (false == is_task_serialized_) {
        COUT << " zmq_task::send perform serialization" << ENDL;
        task_serialized_ = get_serialized();
        is_task_serialized_= true;
      }
      // send data size to prepare memory
      auto size = task_serialized_.size();
      long ret = socket.send((void *)&size, sizeof(size), (flags|ZMQ_SNDMORE) );
      COUT << " zmq_task::send send_size: ret=" << ret << ", sizeof(size)=" << sizeof(size) << ", size=" << size << ENDL;
      // send serialized data
      ret = socket.send(task_serialized_.data(),task_serialized_.size(), flags);
      COUT << " zmq_task::send send_data: ret=" << ret << ", size=" << size << ENDL;
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << local_ids_.size() << "," <<  local_ids_.data() << ") SENT" << ENDL;

    }

    /**
    \brief Receive a task over a ZMQ socket.

    Receive a task over a ZMQ socket.
    \param socket zmq socket
    \param flags sending flags (ZMQ_NOBLOCK)
    */
    void recv(zmq::socket_t &socket, int flags = 0)
    {
      // receive data size and prepare memory
      auto size = task_serialized_.size();
      long ret = socket.recv((void *)&size, sizeof(size), flags);
      COUT << " zmq_task::recv recv_size: ret=" << ret << ", sizeof(size)=" << sizeof(size) << ", size=" << size << ENDL;
      if (ret != sizeof(size)) {
        COUT << "zmq_task::recv zmq_task size recv error" << ENDL;
        throw std::runtime_error("zmq_task::recv zmq_task size recv error");
      }
      task_serialized_.resize(size);
      // receive serialized data
      ret = socket.recv(task_serialized_.data(),task_serialized_.size(), flags);
      COUT << " zmq_task::recv recv_data: ret=" << ret << ", size=" << size << ENDL;
      if (ret != size) {
        COUT << "zmq_task::recv zmq_task recv error" << ENDL;
        throw std::runtime_error("zmq_task::recv zmq_task recv error");
      }
      //deserialize task
      auto aux_serial = std::move(task_serialized_);
      set_serialized(aux_serial.data(), aux_serial.size());
      is_task_serialized_= true;
      task_serialized_ = std::move(aux_serial);
      COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << local_ids_.size() << "," <<  local_ids_.data() << ") RECEIVED" << ENDL;

    }


  private:
    long function_id_;
    long task_id_;
    long order_;
    std::vector<long> local_ids_;
    bool is_hard_;
    std::vector<data_ref_type> data_location_;
    std::set<long> before_dep_;
    std::set<long> after_dep_;

    // serialize state and data
    mutable bool is_task_serialized_;
    mutable std::vector<char> task_serialized_;
    
    friend class boost::serialization::access;

    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned long version)
    {
        if (version >= 0) {
          ar & function_id_;
          ar & task_id_;
          ar & order_;
          ar & local_ids_;
          ar & is_hard_;
          ar & data_location_;
          ar & before_dep_;
          ar & after_dep_;
        }
    }
};

}

#endif

