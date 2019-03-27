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
#ifndef GRPPI_FIFO_SCHEDULER_H
#define GRPPI_FIFO_SCHEDULER_H

#include <vector>
#include <map>

namespace grppi{

constexpr int max_functions = 10000;
constexpr int max_tokens = 100;

template <typename Task>
class fifo_scheduler{
  public:
   // Type alias for task type.
   using task_type = Task;

   /**
   \brief Default construct a fifo scheduler
   */
   fifo_scheduler(){
     functions.reserve(max_functions);
   };

   fifo_scheduler(const fifo_scheduler&) : fifo_scheduler{}
   {
   } 
  
   /**
   \brief Register a new function as sequential task

   Register a new function as a sequential task and initializes the 
   necesary control variables. Sequential task will be executed in 
   series by the thread pool.
   
   \param f Function of the sequential task.
   \return function id for the registered function.
   */
   int register_sequential_task(std::function<void(Task&)> && f)
   {
     return function_id;
   }
  
   /**
   \brief Register a new function as parallel stage in a stream pattern

   Register a new function as a parallel task and initializes the 
   necesary control variables. Multiple task of the registered 
   function may be executed concurrently.
   
   \param f Function of the parallel stage.
   \return function id for the registered function.
   */
   int register_parallel_stage(std::function<void(Task&)> && f)
   {
     return function_id;
   }

   /**
   \brief Removes the functions from the function lists.

   Remove the registered functions and their control variables.
   */
   void clear_tasks()
   {
     functions.clear();
     functions.reserve(max_functions);
     flags.clear();
     seq_tasks.clear();
   }

   /** 
   \brief Introduces a task into the pending tasks queue.

   Introduces a task from a stream pattern in the pending task queue. 
   \param t new task to be launched.
   */
   void launch_task(Task finished, Task new_task)
   {
      // Publish function will receive the finished task and the new task plus an optional node ID
      publish(finished, new_task);
      if( !new_task.is_sequential( )){
         task_queue.push(new_task);
      }
      /* If the task is sequential it should be registered by the
       corresponding node in the recive function for the subscribers*/

   } 

   /**
   \brief Get a new task.

   Get a task in a blocking mode for those threads that has launched 
   new dependent tasks.

   \return task to be run
   */
   Task get_task()
   {
      if(sequential_tasks.ready()){
        return  get_sequential_task();
      }
      return get_parallel_task();
   }

   /**
   \brief Launches a stream pattern.
  
   Launch the first generation task and wait for the end of the pattern
   execution. 
   */
   void run()
   { 
     while( !gen_end || tokens > 0){
       receive_notification();
     }
     clear_tasks(); 
   }


   /**
   \brief Waits for receiving messages from the publisher-subscriber channel
 
   */
   void receive_notification(){
      Task finished, new_task;
      // This function will recieve the messages from other nodes 
      receive(finished, new_task);
      // Check for dependencies
      for( auto & i : dependent_tasks){
         if(i.second.find(finished)){
            i.second.remove(finished);
         }
      }
      /* In charge function will return true if the current node is in char of 
      computing that stage*/
      if( new_task.is_sequential() && in_charge_of(new_task) ){
        sequential_tasks.add(new_task);
      }
   }

  public:
   std::vector<std::function<void(Task&)>> functions;
  private:
   // Local variables of the global scheduler state
   std::atomic<bool> gen_end{true};
   std::atomic<int> tokens{0};
   // Sequential tasks queues per sequential stage 

   // Dependecies:
   // This map stores a tuple of Task - Dependent task IDs
   std::vector<std::tuple<Task, std::vector<int>> dependent_tasks;

   // Distributed task queue 
   
   // Publisher-subscriber communication channel

   
};

}
#endif
