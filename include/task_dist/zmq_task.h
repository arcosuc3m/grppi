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

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
//#pragma GCC diagnostic pop

#include "zmq_data_reference.h"

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
    zmq_task(): function_id_{-1}, task_id_{-1}, order_{-1}, local_ids_{}, is_hard_{false}, data_location_{} {
    COUT << "CREATE TASK: (-1,-1,-1,0,false,-1,-1)" << ENDL;
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
      function_id_{f_id}, task_id_{t_id}, order_{order}, local_ids_{local_ids}, is_hard_{is_hard}, data_location_{} {
      COUT << "CREATE TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << "," << local_ids_.size() << "," << is_hard_ << "," <<  data_location_.size() << ")" << ENDL;

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
      function_id_{f_id}, task_id_{t_id}, order_{order}, local_ids_{local_ids}, is_hard_{is_hard}, data_location_{ref} {
      COUT << "CREATE TASK: ("<< function_id_ << ","<< task_id_ << "," << order_ << "," << local_ids_.size() << "," << is_hard_ << "," <<  data_location_.size() << "," <<  data_location_[0].get_id() << "," << data_location_[0].get_pos() << ")" << ENDL;

      }

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
    std::vector<data_ref_type> get_data_location()
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
    }

    /**
    \brief Return the order number of the task.

    \return  order number of the task
    */
    long get_order()
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
    }

    /**
    \brief Return the list of node ids for local execution.

    \return list of node ids for local execution
    */
    std::vector<long> get_local_ids()
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
    }

    /**
    \brief Return the flag:can only execute on local nodes.

    \return flag:can only execute on local nodes
    */
    bool get_is_hard()
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
    }

    /**
    \brief Return the before dependencies of the task.
    \return before dependencies of the task.
    */
    std::set<long> get_before_dep()
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
    }

    /**
    \brief Return the after dependencies of the task.
    \return after dependencies of the task.
    */
    std::set<long> get_after_dep()
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
    }

    /**
    \brief Construct a task from the serialize string.

    Construct a task from the serialize string.
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
    \brief Get the serialize string for the task.
    
    Get the serialize string for the task.
    \return task serialize string
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
    long function_id_;
    long task_id_;
    long order_;
    std::vector<long> local_ids_;
    bool is_hard_;
    std::vector<data_ref_type> data_location_;
    std::set<long> before_dep_;
    std::set<long> after_dep_;

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

