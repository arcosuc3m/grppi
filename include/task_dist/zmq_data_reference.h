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
    zmq_data_reference(): server_id_{-1}, pos_{-1} {}
    zmq_data_reference(long server_id, long pos) :
        server_id_(server_id), pos_(pos)
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
    \brief Construct a data reference from the serialize string.

    Construct a data reference from the serialize string.
    \param str_data serialize string data
    \param str_size serialize string size
    */
    void set_serialized_string(char * str_data, long str_size)
    {
      boost::iostreams::basic_array_source<char> device(str_data, str_size);
      boost::iostreams::stream<boost::iostreams::basic_array_source<char> > is(device);
      boost::archive::binary_iarchive ia(is);
      try {
        ia >> (*this);
      } catch (...) {
        throw std::runtime_error("Type not serializable");
      }
    }
    
    /**
    \brief Get the serialize string for the data reference.
    
    Get the serialize string for the data reference.
    \return data reference serialize string
    */
    std::string get_serialized_string()
    {
      std::string serial_str;
      boost::iostreams::back_insert_device<std::string> inserter(serial_str);
      boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
      boost::archive::binary_oarchive oa(os);
      try {
        oa << (*this);
        os.flush();
      } catch (...) {
        throw std::runtime_error("Type not serializable");
      }
      return serial_str;
    }

private:
    long server_id_;
    long pos_;

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
