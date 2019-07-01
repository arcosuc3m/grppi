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

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
//#pragma GCC diagnostic pop

#include "zmq_data_reference.h"

namespace grppi{

/**
  \brief Distributed definition for a task.
  This type defines a task as the id of the function to be applied,
  an id for the task and its dependencies.

  This type is used as an example of a task type.
*/

class zmq_task{
  public:

    /**
    \brief Construct an empty task.

    Creates a task with the end function and task ids.
    */
    zmq_task(): function_id_{-1}, task_id_{-1}, data_location_{-1,-1} {
      std::cout << "CREATE TASK: (-1,-1,-1,-1)" << std::endl;
    };

    /**
    \brief Construct a task.

    Creates a task with the function and task ids received as parameters.

    \param f_id function id
    \param t_id task id
    */
    zmq_task(int f_id, long t_id): function_id_{f_id}, task_id_{t_id}, data_location_{-1,-1} {
        {std::ostringstream foo;
         foo << "CREATE TASK: ("<< f_id << ","<< t_id << ",-1,-1)" << std::endl;
        std::cout << foo.str();}
    };
   
    /**
    \brief Construct a task.

    Creates a task with the function and task ids received as parameters.

    \param f_id function id
    \param t_id task id
    \param data_location data location reference
    */
    zmq_task(int f_id, long t_id, zmq_data_reference ref):
      function_id_{f_id}, task_id_{t_id}, data_location_{ref} {
      {std::ostringstream foo;
      foo << "CREATE TASK: ("<< f_id << ","<< t_id << "," << ref.get_id() <<"," << ref.get_pos() << ")" << std::endl;
      std::cout << foo.str();}

      }

    inline bool operator==(const zmq_task& rhs) const {
      return function_id_ == rhs.function_id_;
    }

    inline bool operator!=(const zmq_task& rhs) const {
      return function_id_ != rhs.function_id_;
    } 

    inline bool operator>(const zmq_task& rhs) const {
      return function_id_ > rhs.function_id_;
    }

    inline bool operator<(const zmq_task& rhs) const {
      return function_id_ < rhs.function_id_;
    }
    
    /**
    \brief Return the id of the task function.
    
    Return the id of the task function
    */
    int get_id() const
    {
      return function_id_;
    }

    /**
    \brief Return the task id.

    Return the task id.
    */
    int get_task_id() const
    {
      return task_id_;
    }
    
    /** 
    \brief Return the input data location of the task.

    Return the input data location of the task.
    */
    zmq_data_reference get_data_location()
    {
      return data_location_;
    } 

    /** 
    \brief Store the input data location of the task.

    Store the input data location of the task.
    */
    void set_data_location(zmq_data_reference loc)
    {
      data_location_ = loc;
    }

  private:
    int function_id_;
    long task_id_;
    zmq_data_reference data_location_;

    friend class boost::serialization::access;

    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        if (version >= 0) {
          ar & function_id_;
          ar & task_id_;
          ar & data_location_;
        }
    }
};

}

#endif

