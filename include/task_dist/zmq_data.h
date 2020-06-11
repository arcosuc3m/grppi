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
#ifndef GRPPI_ZMQ_DATA_H
#define GRPPI_ZMQ_DATA_H

#include <iostream>
#include <sstream>
#include <vector>
#include <memory>

#include <zmq.hpp>

#include <boost/any.hpp>

#include "zmq_serialization.h"

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace internal {

/**
\brief Serialize the content of a shared ptr presented as 'any'.

Serialize the content of a shared ptr presented as 'any'
\param item shared ptr to serialize presented as 'any'
\return the char vector with the serialized version of the shared ptr content
*/
template <typename T>
std::vector<char> serialize_any_ptr(const boost::any &item)
{
    return serialize(*(boost::any_cast<std::shared_ptr<T>>(item)));
}

/**
\brief Deserialize an item and store into a shared ptr presented as 'any'.

Deserialize an item and store into a shared ptr presented as 'any'
\param data the serialized version of the item
\param size size of the serialized version of the item
\return shared ptr (presented as 'any') containing the deserialized item
*/
template <typename T>
boost::any deserialize_any_ptr(const char *data, long size)
{
    return std::make_shared<T>{deserialize<T>(data,size)};
}


} // end grppi internal

namespace grppi{

/**
  \brief Distributed definition for a task.
  This type defines a task as the id of the function to be applied,
  an id for the task and its dependencies.

  This type is used as an example of a task type.
*/
class zmq_data{
  public:


    /**
    \brief Construct an empty data.

    Creates an empty data.
    */
    zmq_data(): serialize_data_{},
                ptr_any_data_{},
                serialize_function_{},
                deserialize_function_{};

    /**
    \brief Construct data from copied/moved item.

    Creates data from copied/moved item with serialization function
    \param item item to be copied
    */
    template <typename TItem>
    zmq_data(TItem && item):
        serialize_data_{},
        ptr_any_data_{std::make_shared<TItem>(std::forward<TItem>(item))},
        serialize_function_{internal::serialize_any_ptr<TItem>} {},
        deserialize_function_{internal::deserialize_any_ptr<TItem>};
  
    inline bool operator==(const zmq_data& data) const {
      return (*ptr_any_data_) == data.(*ptr_any_data_); //????
    }

    inline bool operator!=(const zmq_data& data) const {
      return (*ptr_any_data_) != data.(*ptr_any_data_); //????
    }

    /**
    \brief Assign data from copied/moved item.

    Assign data from copied/moved item with serialization function
    \param item item to be copied/moved
    */
    template <typename TItem>
    zmq_data& operator = (TItem && item) {
      serialize_data_.resize(0);
      ptr_any_data_ = std::make_shared<TItem>(std::forward<TItem>(item));
      serialize_function_ = internal::serialize_any_ptr<TItem>;
      deserialize_function_ = internal::deserialize_any_ptr<TItem>;
    }

    /**
    \brief Assign item from copied data.

    Assign item from copied data
    \param data data to be copied
    */
    template <typename TItem>
    TItem& operator = (const zmq_data & item) {
      try {
        // set up serialize funtions if they are not
        if ( (serialize_function_ == nullptr) ||
             (deserialize_function_ == nullptr) ) {
          serialize_function_ = internal::serialize_any_ptr<TItem>;
          deserialize_function_ = internal::deserialize_any_ptr<TItem>;
        }
        // get item stored
        TItem ret;
        if (ptr_any_data_.empty() && serialize_data_.empty()) {
          // not data of any kind, return empty constructed type
          ret = TItem{};
        } else if ( ! ptr_any_data_.empty()) {
          // if real data is not empty, return a copy of it.
          ptr_any_data_ = std::make_shared<TItem>(
                            deserialize_function_(serialize_data_.data(),
                                                 serialize_data_.size()));
          ret = ptr_any_data_;
        } else {
          // real data full, return it
          ret = ptr_any_data_;
        }
        // return item
        return ret;

      } catch (...) {
        std::throw_with_nested( std::runtime_error("zmq_data::operator= Incorrect Type"));
      }
    }

    /**
    \brief Assign item from moved  data.

    Assign item from moved data
    \param data data to be moved
    */
    template <typename TItem>
    TItem& operator = (zmq_data && item) {
      try {
        // set up serialize funtions if they are not
        if ( (serialize_function_ == nullptr) ||
             (deserialize_function_ == nullptr) ) {
          serialize_function_ = internal::serialize_any_ptr<TItem>;
          deserialize_function_ = internal::deserialize_any_ptr<TItem>;
        }
        // get item stored
        TItem ret;
        if (ptr_any_data_.empty() && serialize_data_.empty()) {
          // not data of any kind, return empty constructed type
          ret = TItem{};
        } else if ( ! ptr_any_data_.empty()) {
          // if real data is not empty, return a copy of it.
          ret = std::make_shared<TItem>(
                            deserialize_function_(serialize_data_.data(),
                                                 serialize_data_.size()));
        } else {
          // real data full, return it
          ret = std::move(ptr_any_data_);
        }
        // clean everything
        serialize_data_.resize(0);
        ptr_any_data_ = boost::any{};
        serialize_function_ = nullptr;
        deserialize_function_ = nullptr;
        
        // return item
        return ret;
        
      } catch (...) {
        std::throw_with_nested( std::runtime_error("zmq_data::operator= Incorrect Type"));
      }
    }
    
    /**
    \brief Send data over a ZMQ socket.

    Send data over a ZMQ socket.
    \param socket zmq socket
    \param flags sending flags (ZMQ_SNDMORE,ZMQ_NOBLOCK)
    */
    void send(zmq::socket_t &socket, int flags = 0) const
    {
      try {
        // if not done, serialize data
        if ( (! ptr_any_data_.empty()) && (serialize_data_.empty()) ) {
          COUT << " zmq_data::send perform serialization" << ENDL;
          serialize_data_ = serialize_function_(ptr_any_data_);
        }
        // send data size to prepare memory
        auto size = serialize_data_.size();
        if (size == 0) {
          long ret = socket.send((void *)&size, sizeof(size), flags);
          COUT << " zmq_data::send send_size: ret=" << ret << ", sizeof(size)=" << sizeof(size) << ", size=" << size << ENDL;
        } else {
          long ret = socket.send((void *)&size, sizeof(size), (flags|ZMQ_SNDMORE));
          COUT << " zmq_data::send send_size: ret=" << ret << ", sizeof(size)=" << sizeof(size) << ", size=" << size << ENDL;
          // send serialized data
          ret = socket.send(task_serialized_.data(),task_serialized_.size(), flags);
          COUT << " zmq_data::send send_data: ret=" << ret << ", size=" << size << ENDL;
          COUT << "TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << ") (" << local_ids_.size() << "," <<  local_ids_.data() << ") SENT" << ENDL;
        }
      } catch (..) {
        std::throw_with_nested( std::runtime_error("zmq_data::send Error sending data"));
      }
    }

    /**
    \brief Receive data over a ZMQ socket.

    Receive data over a ZMQ socket.
    \param socket zmq socket
    \param flags sending flags (ZMQ_NOBLOCK)
    */
    void recv(zmq::socket_t &socket, int flags = 0)
    {
      try {
        // receive data size and prepare memory
        auto size = serialize_data_.size();
        long ret = socket.recv((void *)&size, sizeof(size), flags);
        COUT << " zmq_data::recv recv_size: ret=" << ret << ", sizeof(size)=" << sizeof(size) << ", size=" << size << ENDL;
        if (ret != sizeof(size)) {
          COUT << "zmq_data::recv zmq_task size recv error" << ENDL;
          throw std::runtime_error("zmq_task::recv zmq_task size recv error");
        }
        task_serialized_.resize(size);
        if (size == 0) {
          // if size == 0 clean up the data object
          ptr_any_data_ = boost::any{};
          serialize_function_ = nullptr;
          deserialize_function_ = nullptr;
        } else {
          // receive serialized data
          ret = socket.recv(task_serialized_.data(),task_serialized_.size(), flags);
          COUT << " zmq_data::recv recv_data: ret=" << ret << ", size=" << size << ENDL;
          if (ret != size) {
            COUT << "zmq_data::recv zmq_task recv error" << ENDL;
            throw std::runtime_error("zmq_task::recv zmq_task recv error");
          }
        }
      } catch (..) {
        std::throw_with_nested( std::runtime_error("zmq_data::send Error receiving data"));
      }
    }
    
  private:
    std::vector<char>  serialize_data_;
    boost::any ptr_any_data_;
    std::function<std::vector<char>(const boost::any &)> serialize_function_;
    std::function<boost::any(const char *, long)> deserialize_function_;

};

}

#endif

