#include <vector>
#include <map>

namespace grppi{

template <typename Task>
class fifo_scheduler{
  public:
   // Type alias for task type.
   using task_type = Task;

   fifo_scheduler(){};

   fifo_scheduler(const fifo_scheduler&)
   {}

   void register_sequential_task(std::function<void(Task)> && f)
   {
     Task t{(int)functions.size()};
     functions.push_back(f);
     flags.emplace(t, 0);
     seq_tasks.emplace(t, 0);
   }
  
   void register_parallel_stage(std::function<void(Task)> && f)
   {
     functions.push_back(f);
   }

   void register_data_parallel_task(std::function<void(Task)> && f)
   {
     while(task_gen.test_and_set());
     Task t{(int)functions.size()};
     functions.emplace_back(f),tokens++,launch_task(t);
     task_gen.clear();
   }

   void clear_tasks()
   {
     functions.clear();
     flags.clear();
     seq_tasks.clear();
   }

   void launch_task(Task t)
   {
      if(seq_tasks.find(t.get_id()) != seq_tasks.end()) {
         seq_tasks[t]++;
      }
      else tasks.push(t);
   }
  
   void end_sequential_task(Task t)
   {
     flags[t].clear();
   }

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

   void start_task(){
     running_tasks++;
   }

   void finalize_task(){
     running_tasks--;
   }

   void notify_seq_end(Task t)
   {
     flags[t].clear();
   }

   void notify_consume()
   {
     tokens--;
   }

   void run()
   {
     gen_end=false;
     launch_task(Task{0});
     while( !gen_end || tokens > 0 );
     while( running_tasks > 0);
     clear_tasks(); 
   }
  
   void run_data() 
   {
     while(tokens > 0 || running_tasks > 0);
     clear_tasks();
   }

  public:
   std::vector<std::function<void(Task)>> functions;
  private:
   std::atomic_flag task_gen = ATOMIC_FLAG_INIT; 
   std::atomic<bool> gen_end{true};
   std::atomic<int> tokens{0};
   std::atomic<int> running_tasks{0};
   std::map<Task, std::atomic<int>> seq_tasks;
   std::map<Task, std::atomic_flag> flags;
   locked_mpmc_queue<Task> tasks = locked_mpmc_queue<Task>(100);
};

}
