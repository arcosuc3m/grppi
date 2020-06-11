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

#ifndef GRPPI_ZMQ_DATA_REFERENCE_H
#define GRPPI_ZMQ_DATA_REFERENCE_H

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
//#pragma GCC diagnostic pop

#include "zmq_serialization.h"


#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

/**
\defgroup zmq_data_reference zmq data reference
\brief Data reference support types.
@{
*/


class zmq_data_reference
{

public:
    zmq_data_reference(): server_id_{-1}, pos_{-1}, ref_serialized_{}, is_ref_serialized_{false} {}
    zmq_data_reference(long server_id, long pos) :
        server_id_(server_id), pos_(pos), ref_serialized_{}, is_ref_serialized_{false}
    {}
    long get_id() {return server_id_;}
    long get_pos() {return pos_;}
    
    inline bool operator==(const zmq_data_reference& rhs) const {
      COUT << "zmq_data_reference::operator==" << ENDL;
      return (server_id_ == rhs.server_id_) && (pos_ == rhs.pos_);
    }

    inline bool operator!=(const zmq_data_reference& rhs) const {
      COUT << "zmq_data_reference::operator!=" << ENDL;
      return (server_id_ != rhs.server_id_) || (pos_ != rhs.pos_);
    }

    /**
    \brief Construct a data reference from the serialize vector.

    Construct a data reference from the serialize vector.
    \param data serialize data
    \param size serialize size
    */
    void set_serialized(char * data, long size)
    {
      (*this) = internal::deserialize<zmq_data_reference>(data,size);
    }
    
    /**
    \brief Get the serialize vector for the data reference.
    
    Get the serialize vector for the data reference.
    \return data reference serialize vectoir
    */
    std::vector<char> get_serialized()
    {
      return internal::serialize((*this));
    }

    /**
    \brief Send a data reference over a ZMQ socket.

    Send a data reference over a ZMQ socket.
    \param socket zmq socket
    \param flags sending flags (ZMQ_SNDMORE,ZMQ_NOBLOCK)
    */
    void send(zmq::socket_t &socket, int flags = 0)
    {
      // if not done, serialize task
      if (false == is_ref_serialized_) {
        COUT << " zmq_data_reference::send perform serialization" << ENDL;
        ref_serialized_ = get_serialized();
        is_ref_serialized_= true;
      }
      // send data size to prepare memory
      auto size = ref_serialized_.size();
      long ret = socket.send((void *)&size, sizeof(size), (flags|ZMQ_SNDMORE) );
      COUT << " zmq_data_reference::send send_size: ret=" << ret << ", sizeof(size)=" << sizeof(size) << ", size=" << size << ENDL;
      // send serialized data
      ret = socket.send(ref_serialized_.data(),ref_serialized_.size(), flags);
      COUT << " zmq_data_reference::send send_data: ret=" << ret << ", size=" << size << ENDL;

    }


    /**
    \brief Receive a data reference over a ZMQ socket.

    Receive a data reference over a ZMQ socket.
    \param socket zmq socket
    \param flags sending flags (ZMQ_NOBLOCK)
    */
    void recv(zmq::socket_t &socket, int flags = 0)
    {
      // receive data size and prepare memory
      auto size = ref_serialized_.size();
      long ret = socket.recv((void *)&size, sizeof(size), flags);
      COUT << " zmq_data_reference::recv recv_size: ret=" << ret << ", sizeof(size)=" << sizeof(size) << ", size=" << size << ENDL;

      if (ret != sizeof(size)) {
        throw std::runtime_error("zmq_data_reference::recv zmq_data_reference size recv error");
      }
      ref_serialized_.resize(size);
      // receive serialized data
      ret = socket.recv(ref_serialized_.data(),ref_serialized_.size(), flags);
      COUT << " zmq_data_reference::recv recv_data: ret=" << ret << ", size=" << size << ENDL;
      if (ret != size) {
        throw std::runtime_error("zmq_data_reference::recv zmq_data_reference recv error");
      }
      //deserialize task
      auto aux_serial = std::move(ref_serialized_);
      set_serialized(aux_serial.data(), aux_serial.size());
      is_ref_serialized_= true;
      ref_serialized_ = std::move(aux_serial);
    }

private:
    long server_id_;
    long pos_;

    bool is_ref_serialized_=false;
    std::vector<char> ref_serialized_;
    
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned long version)
    {
        if (version >= 0) {
          ar & server_id_;
          ar & pos_;
        }
    }
};

/**
@}
*/


}

#endif
