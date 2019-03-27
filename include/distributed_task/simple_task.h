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
#ifndef GRPPI_SIMPLE_TASK_H
#define GRPPI_SIMPLE_TASK_H

#include <set>

namespace grppi{

/**
  \brief Simple definition for a task.
  This type defines a task as the id of the function to be applied,
  an id for the task and its dependencies.

  This type is used as an example of a task type.
*/

class simple_task{
  public:

   /**
   \brief Construct an empty task.

   Creates a task with the end function and task ids.
   */
   simple_task(): function_id_{-1}, task_id_{-1} {};

   /**
   \brief Construct a task.

   Creates a task with the function and task ids received as parameters.

   \param f_id function id
   \param t_id task id
   */
   simple_task(int f_id, long t_id): function_id_{f_id}, task_id_{t_id} {};

   inline bool operator==(const simple_task& rhs) const {
      return function_id_ == rhs.function_id_;
    }

    inline bool operator!=(const simple_task& rhs) const {
      return function_id_ != rhs.function_id_;
    } 

    inline bool operator>(const simple_task& rhs) const {
      return function_id_ > rhs.function_id_;
    }

    inline bool operator<(const simple_task& rhs) const {
      return function_id_ < rhs.function_id_;
    }
    
    /**
    \brief Return the id of the pattern.

    If a task is part of a given pattern, returns its id.
    */
    int get_pattern_id() const
    {
      return pattern_;
    }
   
    /**
    \brief Set the id of the pattern.

    Set the value of the pattern id to the id received as parameter.
    \param p_id Pattern id.
    */
    void set_pattern_id(int p_id)
    {
      pattern_ = p_id;
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
    \brief Return the input data location of the task.

    Return the input data location of the task.
    */
    int get_input_data_location()
    {
      return intpu_data_location_;
    } 

    /** 
    \brief Store the input data location of the task.

    Store the input data location of the task.
    */
    void set_input_data_location(int loc)
    {
      input_data_location_ = loc;
    } 

    /** 
    \brief Return the input data location of the task.

    Return the input data location of the task.
    */
    int get_output_data_location()
    {
      return output_data_location_;
    } 

    /** 
    \brief Store the input data location of the task.

    Store the input data location of the task.
    */
    void set_output_data_location(int loc)
    {
      output_data_location_ = loc;
    } 


    /** 
    \brief Return the task id.

    Return the task id.
    */
    int get_task_id() const
    {
      return task_id_;
    }

  private:
    int function_id_;
    long task_id_;
    int input_data_location_;
    int output_data_location_;
    int pattern_ = -1;
};

}

#endif

