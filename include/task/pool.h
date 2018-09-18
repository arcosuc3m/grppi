#include <thread>
#include <iostream>
#include <vector>
#include "mpmc_queue.h"
#include <functional>
#include <optional>
#include <atomic>
#include <map>
#include <tuple>
#include <memory>

namespace grppi{

class pool
{
  public:

    pool(int pool_size){
      for (int i=0; i<pool_size; i++){
          pool_threads.push_back(
             std::thread([this](){
                while(1){
                  int id = get_task();
                  if( id == -1) break;
                  functions[id]();
                  if(!gen_end && id == 0){
                    launch_task(0);
                  }
                  id++;
                  launch_task(id);
                }
             })
          );
       }
    }

    void finalize_pool(){
       for(int i=0; i < pool_threads.size(); i++){
          tasks.push(-1);        
       }
       for(int i=0; i < pool_threads.size(); i++){
          pool_threads[i].join();
       }
    }

    void launch_task(int task_id)
    { 
       if(task_id < functions.size()){
          if(seq_tasks.find(task_id) != seq_tasks.end()) seq_tasks[task_id]++;
          else tasks.push(task_id);
       }
       if(seq_tasks.find(task_id-1) != seq_tasks.end()) flags[task_id-1].clear();
    }

    void add_sequential_task(std::function<void()> && f){
       flags.emplace(functions.size(), 0);
       seq_tasks.emplace(functions.size(), 0);
       functions.push_back(f);
    }

    void add_parallel_task(std::function<void()> && f){
       functions.push_back(f);
    }

    void clear_tasks(){
       functions.clear();
       flags.clear();
       seq_tasks.clear();
    }

    void run_data(){
       for(int i=0;i < functions.size()-1;i++){
	  tokens++;
          launch_task(i);
       } 
       functions[functions.size()-1]();
       while(tokens>0);
       clear_tasks();
    }

    void run(){
       launch_task(0);
       //TODO: Modify this loop for using a condition variable
       while((!gen_end && !pipe_end) || tokens > 0);
       clear_tasks();
    }

  private:
    int get_task()
    {
       while(tokens> 0 || !pipe_end){
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

  public:
    std::atomic<bool> pipe_end = false;
    std::atomic<bool> gen_end = false;
    std::atomic<int> tokens = 0;

  private:
    std::map<int, std::atomic<int>> seq_tasks;
    std::map<int, std::atomic_flag> flags;
    locked_mpmc_queue<int> tasks = locked_mpmc_queue<int>(100);
    //TODO: include tasck-dependency control mechanism. Need to define a 
    //      new object for storing the context of a waiting task.
    std::vector<std::function<void()>> functions;
    std::vector<std::thread> pool_threads;
};

}
