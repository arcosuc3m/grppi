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
#include <memory>

namespace grppi{

constexpr int max_functions = 10000;
constexpr int max_tokens = 100;

template <typename Task>
class zmq_scheduler{
  public:
   // Type alias for task type.
   using task_type = Task;

   /**
   \brief Default construct a fifo scheduler
   */
   zmq_scheduler(){
     functions.reserve(max_functions);
     data_service_ = make_shared<zmq_data_service>();
   };

   zmq_scheduler(const zmq_scheduler&) : zmq_scheduler{}
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
     Task t{(int)functions.size(), 0};
     int function_id = (int) functions.size();
     functions.push_back(f);
     flags.emplace(t, 0);
     seq_tasks.emplace(t, 0);
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
     while(task_gen.test_and_set());
     int function_id = (int) functions.size();
     functions.emplace_back(f);
     task_gen.clear();
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
   void launch_task(Task t)
   {
      if(seq_tasks.find(t) != seq_tasks.end()) {
         seq_tasks[t]++;
      }
      else {
        tasks.push(t);
      }
   }
   
   /**
   \brief Get a new task.

   Get a task in a blocking mode for those threads that has launched 
   new dependent tasks.

   \return task to be run
   */
   Task get_task()
   {
      while(tokens> 0 || !gen_end){
        for(auto &i: flags)
        {
          if(!i.second.test_and_set()){
            if(seq_tasks[i.first]>0) {
              seq_tasks[i.first]--;
              return i.first;
            }
            i.second.clear();
          }
        }
        if(!tasks.empty()) return tasks.pop();
      }
      return tasks.pop();
   }

   /**
   \brief Notifies the scheduler that a new task has started its execution.
   
   Notifies the scheduler that a new task has started its execution.
   */
   void start_task(){
     running_tasks++;
   }

   /**
   \brief Notifies the scheduler that a task has finished.
   
   Notifies the scheduler that a new task has finished its execution.

   \param t finished task.
   */
   void finalize_task(Task t){
     running_tasks--;
   }

   /**
   \brief Notifies the finalization of a sequential stage.
   
   Notifies the finalization of a sequential stage in stream patterns.
   \param t finished task
   */
   void notify_sequential_end(Task t)
   {
     flags[t].clear();
   }

   /**
   \brief Notifies the consumption of an item in stream patterns.
   
   Notifies the consumption of an item in stream patterns.
   */
   void notify_consumer_end()
   {
     tokens--;
   }

   /**
   \brief Launches a stream pattern.
  
   Launch the first generation task and wait for the end of the pattern
   execution. 
   */
   void run()
   {
     gen_end=false;
     launch_task(Task{0, 0});
     while( !gen_end || tokens > 0 || running_tasks > 0);
     clear_tasks(); 
   }

   /**
   \brief Notifies the generation of a new token

   Notifies the generation of a new token
   */
   void new_token(){
      tokens++;
   }

   /** 
   \brief Notifies the end of the generation stage

   Notifies the end of the generation stage
   */
   void pipe_stop(){
      gen_end=true;
   }
   
   
  /**
  \brief Get the data element from the server and position
  referenced in the ref param.
  \tparam T Element type for the data element.
  \param ref zmq_reference of the server and position for tha data.
  */
  template <class T>
  T get (zmq_reference ref)
  {
    return data_service->get(ref);
  }

  /**
  \brief Get the data element from the server and position
  referenced in the ref param.
  \tparam T Element type for the data element.
  \param elem element to store at the data server.
  \param ref zmq_reference of the server and position for tha data.
  */
  template <class T>
  zmq_reference set(T item)
  {
      return data_service->set(ref);
  }
  
  
  public:
   std::vector<std::function<void(Task&)>> functions;
  private:
   std::atomic_flag task_gen = ATOMIC_FLAG_INIT;
   std::atomic<bool> gen_end{true};
   std::atomic<int> tokens{0};
   std::atomic<int> running_tasks{0};
   std::map<Task, std::atomic<int>> seq_tasks;
   std::map<Task, std::atomic_flag> flags;
   locked_mpmc_queue<Task> tasks = locked_mpmc_queue<Task>(max_tokens);
   std::shared_ptr<zmq_data_service> data_service = std::shared_ptr<zmq_data_service>();
}
#endif
