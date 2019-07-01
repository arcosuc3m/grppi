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

namespace grppi{

/**
\defgroup zmq_data_reference zmq data reference
\brief Data reference support types.
@{
*/


class zmq_data_reference
{
private:
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        if (version >= 0) {
          ar & server_id_;
          ar & pos_;
        }
    }
    int server_id_;
    int pos_;
public:
    zmq_data_reference() {}
    zmq_data_reference(int server_id, int pos) :
        server_id_(server_id), pos_(pos)
    {}
    int get_id() {return server_id_;}
    int get_pos() {return pos_;}
};

/**
@}
*/


}

#endif
