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
   \brief Register a new function as parallel task in data patterns

   Register a new function as a parallel task and initializes the 
   necesary control variables. Multiple task of the registered 
   function may be executed concurrently.
   
   \param f Function of the parallel task.
   \return function id for the registered function.
   */
   int register_data_parallel_task(std::function<void(Task&)> && f)
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

   Introduces a task from a data pattern in the pending task queue. 
   It also increments the number of active tokens and the number of tokens 
   related to the pattern.

   \param t new task to be launched.
   */
   void launch_data_task(Task t)
   {
     tokens++;
     patterns_dep[t.get_pattern_id()]++;
     tasks.push(t);
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
   \brief Notifies the end of a sequential task.

   Notifies the scheduler that a sequential task has finished.
  
   \param t finished task.
   */
   void end_sequential_task(Task t)
   {
     flags[t].clear();
   }

   /**
   \brief Get a new task.

   Get a task in a non-blocking mode for those threads that has launched 
   new dependent tasks.
   
   \param task Reference to store the obtained task.
   \return Boolean that indicates if the function obtained a task to be run.
   */
   bool try_get_task(Task& task)
   {
     while(tokens> 0 || !gen_end){
       for(auto &i: flags)
       {
         if(!i.second.test_and_set()){
           if(seq_tasks[i.first]>0) {
             seq_tasks[i.first]--;
             task = i.first;
             return true;
           }
           i.second.clear();
         }
       }
       return tasks.try_pop(task);
     }
     return tasks.try_pop(task);

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
   This function also updates the task dependencies.

   \param t finished task.
   */
   void finalize_task(Task t){
     // Resolve dependencies
     while(pattern_instantiation.test_and_set());
     if( t.get_pattern_id() != -1 ) patterns_dep[t.get_pattern_id()]--;
     pattern_instantiation.clear();

     for( auto it = t.after_dependencies_.begin(); it != t.after_dependencies_.end(); it++)
     { 
       bool done = false;
       while(!done)
       {
         for( int i = 0; i<max_tokens; i++) {
           while(wait[i].test_and_set());
           if( used[i] != 0){
             if ( waiting_tasks[i].get_task_id() == *it){
               waiting_tasks[i].before_dependencies_.erase(t.get_task_id());
               if(waiting_tasks[i].before_dependencies_.size() == 0) {
                 tasks.push(waiting_tasks[i]);
                 used[i] = 0;
               }
               done = true; 
               wait[i].clear();
               break;
             }
           }
           wait[i].clear();
         }     
       }
     }
     running_tasks--;
   }

   /**
   \brief Store a task in the waiting task queue.
   
   Stores a task that has dependencies in the waiting task queue.

   \param t dependent task.
   */
   void wait_for_dependencies(Task t)
   {
     tokens++;
     bool introduced=false;
     //Check for an empty position
     while(!introduced){
       for(int i = 0; i<max_tokens; i++){
         if(!wait[i].test_and_set())
         {
           if(used[i] == 0)
           {
             used[i]=1;
             introduced=true;
             waiting_tasks[i] = t;
             wait[i].clear();
             break;
           }
           wait[i].clear();
         }
       }
     }
   }

   /**
   \brief Notifies the finalization of a sequential stage.
   
   Notifies the finalization of a sequential stage in stream patterns.
   \param t finished task
   */
   void notify_seq_end(Task t)
   {
     flags[t].clear();
   }

   /**
   \brief Notifies the consumption of an item in stream patterns.
   
   Notifies the consumption of an item in stream patterns.
   */
   void notify_consume()
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
     launch_task(Task{0, task_id++});
     while( !gen_end || tokens > 0 || running_tasks > 0);
     clear_tasks(); 
   }

   /**
   \brief Launches a data pattern.
  
   Wait for the end of the pattern execution. 

   \note To be deprecated in the next version.
   */
   void run_data() 
   {
      while(tokens > 0 || running_tasks > 0);
      clear_tasks();
   }

   /**
   \brief Notifies the start of a pattern instantiation.
  
   Notifies the start of a pattern instantiation and initializes the 
   necesary control mechanisms.

   \return returns the pattern id.
   */
   int start_pattern() 
   {
     while(pattern_instantiation.test_and_set());
     int aux = pattern_id_++;
     patterns_dep.emplace(aux,0);
     pattern_instantiation.clear();
     return aux;
   }

   /**
   \brief Launches a data pattern.
  
   Wait for the end of the pattern execution with the id received as
   parameter. The waiting concurrent entity is reintroduced into the pool
   to run pending tasks.
  
   \param pattern_id id of the launched pattern.
   */
   void run_data(int pattern_id) 
   {
     while(patterns_dep[pattern_id] > 0 ){
       //TODO: Probably not the best solution
       Task t{-1,-1};
       if( try_get_task(t) && t.get_id() != -1){
         start_task(),
         functions[t.get_id()](t),
         finalize_task(t);
       } 
     }
     while(pattern_instantiation.test_and_set());
     patterns_dep.erase(pattern_id);
     if(patterns_dep.size() == 0 ){
       pattern_id_= 0;
       clear_tasks();
       tokens = 0;
     }
     pattern_instantiation.clear();
   }

   /**
   \brief Obtain a new id for a task.

   Return the current task id and increment it value for the next call.
   \reutnr task id
   */
   long get_task_id(){
     return task_id++;
   }

   /**
   \brief Checks if the scheduler has already generated the maximum 
   number of tokens.
   
   Checks if the scheduler has already generated the maximum number of tokens.
   
   \param tokens that should be generated
   \return boolean indicating if the new tokens can be generated
   */
   bool is_full(int new_tokens)
   {
     return tokens+new_tokens>=max_tokens/2;
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

  public:
   std::vector<std::function<void(Task&)>> functions;
   int chunk_size_ = 0;
  private:
   std::atomic_flag task_gen = ATOMIC_FLAG_INIT; 
   std::atomic<bool> gen_end{true};
   std::atomic<int> tokens{0};
   std::atomic<long> task_id{0};
   std::atomic<int> running_tasks{0};
   std::map<Task, std::atomic<int>> seq_tasks;
   std::map<Task, std::atomic_flag> flags;
   std::atomic_flag already_running{ATOMIC_FLAG_INIT};
   locked_mpmc_queue<Task> tasks = locked_mpmc_queue<Task>(max_tokens);
   
   //Pattern dependencies
   std::atomic_flag pattern_instantiation{ATOMIC_FLAG_INIT};
   int pattern_id_{0};
   std::map< int, std::atomic<int> > patterns_dep;

   // Store waiting tasks
   Task waiting_tasks[max_tokens];
   bool used[max_tokens] = {};
   std::atomic_flag wait[max_tokens] = {ATOMIC_FLAG_INIT};
  

   
};

}
#endif
