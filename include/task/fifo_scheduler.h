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

   fifo_scheduler(){
     functions.reserve(max_functions);
   };

   fifo_scheduler(const fifo_scheduler&){
     functions.reserve(max_functions);
   } 

   int register_sequential_task(std::function<void(Task&)> && f)
   {
     Task t{(int)functions.size(), 0};
     int function_id = (int) functions.size();
     functions.push_back(f);
     flags.emplace(t, 0);
     seq_tasks.emplace(t, 0);
     return function_id;
   }
  
   int register_parallel_stage(std::function<void(Task&)> && f)
   {
     while(task_gen.test_and_set());
     int function_id = (int) functions.size();
     functions.emplace_back(f);
     task_gen.clear();
     return function_id;
   }

   int register_data_parallel_task(std::function<void(Task&)> && f)
   {
     while(task_gen.test_and_set());
     int function_id = (int) functions.size();
     //Task t{(int)functions.size(),task_id++};
     functions.emplace_back(f);/*tokens++,launch_task(t);*/
     task_gen.clear();
     return function_id;
   }

   void clear_tasks()
   {
     functions.clear();
     functions.reserve(max_functions);
     flags.clear();
     seq_tasks.clear();
   }

   void launch_data_task(Task t)
   {
     tokens++;
     patterns_dep[t.get_pattern_id()]++;
     tasks.push(t);
   }

   void launch_task(Task t)
   {
      if(seq_tasks.find(t) != seq_tasks.end()) {
         seq_tasks[t]++;
      }
      else tasks.push(t);
   }
  
   void end_sequential_task(Task t)
   {
     flags[t].clear();
   }

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
     launch_task(Task{0, task_id++});
     while( !gen_end || tokens > 0 );
     while( running_tasks > 0);
     clear_tasks(); 
   }
  
   void run_data() 
   {
      while(tokens > 0 || running_tasks > 0);
      clear_tasks();
   }


   int start_pattern() 
   {
     while(pattern_instantiation.test_and_set());
     int aux = pattern_id_++;
     patterns_dep.emplace(aux,0);
     pattern_instantiation.clear();
     return aux;
   }

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

   long get_task_id(){
     return task_id++;
   }

   bool is_full(int new_tokens)
   {
     return tokens+new_tokens>=max_tokens/2;
   }

   void new_token(){
      tokens++;
   }

   void pipe_stop(){
      gen_end=true;
   }

  public:
   std::vector<std::function<void(Task&)>> functions;
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
