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
#ifndef GRPPI_ZMQ_SCHEDULER_H
#define GRPPI_ZMQ_SCHEDULER_H
#include <new>

#include <vector>
#include <map>
#include <memory>
#include <set>
#include <atomic>
#include <thread>
#include <exception>
#include <assert.h>

#include <zmq.hpp>

#include "../common/mpmc_queue.h"
#include "../common/configuration.h"

#include "dist_pool.h"
#include "zmq_data_reference.h"
#include "zmq_data_service.h"
#include "zmq_port_service.h"
#include "multi_queue.h"

#undef COUT
#define COUT if (0) std::cout

namespace grppi{

#pragma GCC diagnostic warning "-Wunused-parameter"

template <typename Task>
class zmq_scheduler_thread;

template <typename Task>
class zmq_scheduler{
  public:
    // Type alias for task type.
    using task_type = Task;
   
    // no copy constructors
    zmq_scheduler(const zmq_scheduler&) =delete;
    zmq_scheduler& operator=(zmq_scheduler const&) =delete;

    void launch_thread();

    /**
    \brief Default construct a zmq scheduler
    */

    zmq_scheduler(std::map<int, std::string> machines, int id,
              const std::shared_ptr<zmq_port_service> &port_service,
              int numTokens, int server_id) :
      machines_(machines.begin(), machines.end()),
      id_(id),
      server_id_(server_id),
      is_sched_server_(id == server_id),
      schedServer_portserv_port_(port_service->new_port()),
      max_tokens_(numTokens),
      total_servers_(machines_.size()),
      port_service_(port_service),
      context_(1)
    {
    
      functions.reserve(max_functions);
      COUT << "zmq_scheduler: data_service_\n";
      data_service_ = std::make_shared<zmq_data_service>(machines_, id_,
                                                         port_service_, max_tokens_);
      COUT << "zmq_scheduler: data_service_ done\n";

      // if server, bind reply socket and launch thread
      if (is_sched_server_) {
        // server thread launched
        COUT << "zmq_scheduler: launch_thread()\n";
        launch_thread();
        COUT << "zmq_scheduler: launch_thread() done\n";
      }

      // get secheduler server port
      COUT << "zmq_scheduler: port_service_->get \n";
      schedServer_port_ = port_service_->get(0,schedServer_portserv_port_, true);
      COUT << "zmq_scheduler: port_service_->get end \n";
   
      // launch thread pool
      concurrency_degree_= configuration<>{}.concurrency_degree();
      thread_pool_.init(this,concurrency_degree_);

    };

    /**
    \brief Default destrucutor for a zmq scheduler
    */
    ~zmq_scheduler() {
      COUT << "zmq_scheduler::~zmq_scheduler BEGIN\n";
      end();
      thread_pool_.finalize_pool();
      if (is_sched_server_) {
        server_thread_.join();
      }
      COUT << "zmq_scheduler::~zmq_scheduler END\n";
    }


    /**
    \brief Register a new function as sequential task

    Register a new function as a sequential task and initializes the
    necesary control variables. Sequential task will be executed in
    series by the thread pool.
   
    \param f Function of the sequential task.
    \param create_tokens True: task can create more than one tasks.
    \return function id for the registered function.
    */
    int register_sequential_task(std::function<void(Task&)> && f,
                                bool create_tokens)
    {
      while(task_gen.test_and_set());
      int function_id = functions.size();;
      functions.emplace_back(f);
      seq_func_ids_.push_back(function_id);
      if (create_tokens) {
        new_token_func_.push_back(1);
      } else {
        new_token_func_.push_back(0);
      }
      task_gen.clear();
      {std::ostringstream ss;
      ss << "register_sequential_task: func_id=" << function_id << std::endl;
      COUT << ss.str();}
      return function_id;
    }
  
   /**
   \brief Register a new function as parallel stage in a stream pattern

   Register a new function as a parallel task and initializes the 
   necesary control variables. Multiple task of the registered 
   function may be executed concurrently.
   
   \param f Function of the parallel stage.
   \param create_tokens True: task can create more than one tasks.
   \return function id for the registered function.
   */
   int register_parallel_stage(std::function<void(Task&)> && f,
                                bool create_tokens)
   {
     while(task_gen.test_and_set());
     int function_id = (int) functions.size();
     functions.emplace_back(f);
     if (create_tokens) {
       new_token_func_.push_back(1);
     } else {
       new_token_func_.push_back(0);
     }
     task_gen.clear();
     {std::ostringstream ss;
     ss << "register_parallel_stage: func_id=" << function_id << std::endl;
     COUT << ss.str();}
     return function_id;
   }

   /**
   \brief Removes the functions from the function lists.
   Remove the registered functions and their control variables.
   */
   void clear_tasks()
   {
     while(task_gen.test_and_set());
    
     // reset functions vector
     functions.clear();
     functions.reserve(max_functions);

     // reset functions vector
     seq_func_ids_.clear();
     seq_func_ids_.reserve(max_functions);
     task_gen.clear();
   }


   //**********   client part of the server messages tasks
   
   /**
   \brief Introduces a task into the pending tasks queue.

   Introduces a task from a stream pattern in the pending task queue. 
   \param task new task to be launched.
   \param new_token new task needs a new token or not.
   */
   void set_task(Task task, bool new_token)
   {
     try {
       COUT << "zmq_scheduler:set_task BEGIN\n";

       // Get the socket for this thread
       while(accessSockMap.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap.clear();
       COUT << "zmq_scheduler:set_task requestSock_ obtain\n";

       int new_token_int = (new_token ? 1 : 0);
       COUT << "zmq_scheduler:set_task requestSock_ A\n";
       requestSock_->send(setTaskCmd.data(), setTaskCmd.size(), ZMQ_SNDMORE);
       COUT << "zmq_scheduler:set_task requestSock_ B\n";
       requestSock_->send((int *)(&id_), sizeof(id_), ZMQ_SNDMORE);
       COUT << "zmq_scheduler:set_task requestSock_ C\n";
       std::string task_string = task.get_serialized_string();
       requestSock_->send((char *)(task_string.data()), task_string.size(), ZMQ_SNDMORE);
       COUT << "zmq_scheduler:set_task requestSock_ D\n";
       requestSock_->send((int *)(&new_token_int), sizeof(new_token_int));

       {std::ostringstream ss;
       ss << "zmq_scheduler::set_task SENT: id_: "
          << id_ << ", task: (" << task.get_id()
          << ", " << task.get_task_id()
          << "), new_token_int = " << new_token_int << std::endl;
       COUT << ss.str();}

       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       {std::ostringstream ss;
       ss << "zmq_scheduler::set_task: size=" << message.size() << std::endl;
       COUT << ss.str();}

       if (message.size() != 0) {
          COUT << "Error: Wrong ACK\n";
          throw std::runtime_error("ERROR: Wrong ACK");
       }
       COUT << "zmq_scheduler:set_task END\n";

     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:set_task" << e.what() << '\n';
     }
   }
   
   /**
   \brief Get a new task.

   Get a task in a blocking mode for those threads that has launched 
   new dependent tasks.
   \param old_task old task that have been executed.
   \return new task to be run
   */
   Task get_task(Task old_task)
   {
     try {
       COUT << "zmq_scheduler:get_task BEGIN\n";
       // Get the socket for this thread
       while(accessSockMap.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap.clear();
       COUT << "zmq_scheduler:get_task requestSock_ obtain\n";

       //auto old_task_id = old_task.get_task_id();
       requestSock_->send(getTaskCmd.data(), getTaskCmd.size(), ZMQ_SNDMORE);
       requestSock_->send((int *)(&id_), sizeof(id_), ZMQ_SNDMORE);
       std::string task_string = old_task.get_serialized_string();
       requestSock_->send((char *)(task_string.data()), task_string.size());
       
       {std::ostringstream ss;
       ss << "zmq_scheduler::get_task SENT: id_: "
          << id_ << ", old_task: (" << old_task.get_id()
          << ", " << old_task.get_task_id() << ")\n";
       COUT << ss.str();}
       
       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       {std::ostringstream ss;
       ss << "zmq_scheduler::get_task: size=" << message.size() << std::endl;
       COUT << ss.str();}

       if (message.size() == 0) {
          COUT << "Error: get task does not return a TASK\n";
          throw std::runtime_error("ERROR: get task does not return a TASK");
       }
       Task new_task{};
       new_task.set_serialized_string((char *)message.data(),message.size());

       COUT << "zmq_scheduler:get_task END \n";
       return (new_task);
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:get_task" << e.what() << '\n';
       return (Task{});
     }
   }

   /**
   \brief Notifies the consumption of an item in stream patterns.
   
   Notifies the consumption of an item in stream patterns.
   */
   void consume_token()
   {
     try {
       COUT << "zmq_scheduler:consume_token BEGIN \n";
       // Get the socket for this thread
       while(accessSockMap.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap.clear();
       COUT << "zmq_scheduler:consume_token requestSock_ obtain\n";

       requestSock_->send(consumeCmd.data(), consumeCmd.size());
       COUT << "zmq_scheduler:consume_token SENT\n";

       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       {std::ostringstream ss;
       ss << "zmq_scheduler::consume_token: size=" << message.size() << std::endl;
       COUT << ss.str();}

       if (message.size() != 0) {
          COUT << "Error: Wrong ACK\n";
          throw std::runtime_error("ERROR: Wrong ACK");
       }
       COUT << "zmq_scheduler:consume_token END\n";
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:consume_token" << e.what() << '\n';
     }
   }

   /**
   \brief starts a stream pattern form each machine.
  
   Starts a stream pattern form each machine. When all has notified the running
   command it will start, all run functions will block until all is done.
   
   */
   void run()
   {
     try {
       COUT << "zmq_scheduler:run BEGIN\n";
       // Get the socket for this thread
       while(accessSockMap.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap.clear();
       COUT << "zmq_scheduler::run requestSock_ obtain\n";

       int is_sched_server_int = (is_sched_server_ ? 1 : 0);

       requestSock_->send(tskRunCmd.data(), tskRunCmd.size(), ZMQ_SNDMORE);
       requestSock_->send((int *)(&concurrency_degree_), sizeof(concurrency_degree_), ZMQ_SNDMORE);
       requestSock_->send((int *)(&id_), sizeof(id_), ZMQ_SNDMORE);
       if (is_sched_server_) {
           requestSock_->send((int *)(&is_sched_server_int), sizeof(is_sched_server_int), ZMQ_SNDMORE);
           requestSock_->send(seq_func_ids_.data(), seq_func_ids_.size() * sizeof(int), ZMQ_SNDMORE);
           requestSock_->send(new_token_func_.data(), new_token_func_.size() * sizeof(int));

           {std::ostringstream ss;
           ss << "zmq_scheduler::run SENT: concurrency_degree_: "
              << concurrency_degree_ << ", id_: " << id_
              << ", is_sched_server_" << is_sched_server_
              << ", seq_func_ids_.size()" << seq_func_ids_.size()
              << ", new_token_func_.size() = " << new_token_func_.size()
              << std::endl;
           COUT << ss.str();}
       } else {
           requestSock_->send((int *)(&is_sched_server_int), sizeof(is_sched_server_int));

           {std::ostringstream ss;
           ss << "zmq_scheduler::run SENT: concurrency_degree_: "
              << concurrency_degree_ << ", id_: " << id_
              << ", is_sched_server_" << is_sched_server_
              << std::endl;
           COUT << ss.str();}
       }

       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       {std::ostringstream ss;
       ss << "zmq_scheduler::run: size=" << message.size() << std::endl;
       COUT << ss.str();}

       if (message.size() != 0) {
          COUT << "Error: Wrong ACK\n";
          throw std::runtime_error("ERROR: Wrong ACK");
       }
       
       // clean task arrays for next execution
       clear_tasks();
       COUT << "zmq_scheduler:run END\n";
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:run" << e.what() << '\n';
     }
   }

   /**
   \brief ends the scheduler and scheduler thread.
  
   Sends a message to the scheduler thread to end and to end the working threads.
   
   */
   void end()
   {
     try {
       COUT << "zmq_scheduler:end BEGIN\n";
       // Get the socket for this thread
       while(accessSockMap.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap.clear();
       COUT << "zmq_scheduler::end requestSock_ obtain\n";

       requestSock_->send(tskEndCmd.data(), tskEndCmd.size(), ZMQ_SNDMORE);
       requestSock_->send((int *)(&id_), sizeof(id_));
       
       {std::ostringstream ss;
       ss << "zmq_scheduler::end SENT: id_: "
          << id_  << std::endl;
       COUT << ss.str();}
       
       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       {std::ostringstream ss;
       ss << "zmq_scheduler::end: size=" << message.size() << std::endl;
       COUT << ss.str();}

       if (message.size() != 0) {
          COUT << "Error: Wrong ACK\n";
          throw std::runtime_error("ERROR: Wrong ACK");
       }
       COUT << "zmq_scheduler:end END\n";
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:end" << e.what() << '\n';
     }
   }
   //**********  END client part of the server messages tasks


  /**
  \brief Get the data element from the server and position
  referenced in the ref param.
  \tparam T Element type for the data element.
  \param ref zmq_data_reference of the server and position for tha data.
  */
  template <class T>
  T get (zmq_data_reference ref)
  {
    return data_service_->get<T>(ref);
  }

  /**
  \brief Get the data element from the server and position
  referenced in the ref param.
  \tparam T Element type for the data element.
  \param elem element to store at the data server.
  \param ref zmq_data_reference of the server and position for tha data.
  */
  template <class T>
  zmq_data_reference set(T item)
  {
      return data_service_->set(item);
  }
  
  
  /**
  \brief Set number of grppi threads.
  */
  void set_concurrency_degree(int degree) noexcept { concurrency_degree_ = degree; }

  /**
  \brief Get number of grppi threads.
  */
  int concurrency_degree() const noexcept { return concurrency_degree_; }


  public:
    // collection of stage functions
    std::vector<std::function<void(Task&)>> functions;
    std::vector<int> max_num_new_tokens;

    template <typename T>
    friend class grppi::zmq_scheduler_thread;

  private:
    /// tcp bind pattern
    const std::vector<std::string> tcpBindPattern {"tcp://*:", ""};
    /// tcp connect pattern
    const std::vector<std::string> tcpConnectPattern {"tcp://", ":"};

    /// tag for task end command
    const std::string tskEndCmd{"END_SCH"};
    /// tag for task run command
    const std::string tskRunCmd{"RUN_PIP"};
    /// tag for get task command
    const std::string getTaskCmd{"GET_TSK"};
    /// tag for set task command
    const std::string setTaskCmd{"SET_TSK"};
    /// tag for consume command
    const std::string consumeCmd{"CONSUME"};


    constexpr static int default_functions_ = 10000;
    constexpr static int default_tokens_ = 100;

    const int max_functions = default_functions_;
    const int max_tokens = default_tokens_;
    

    // construct params
    std::map<int, std::string> machines_;
    int id_{0};
    int server_id_{0};
    int schedServer_port_{0};
    bool is_sched_server_{false};
    int schedServer_portserv_port_{1};
    int max_tokens_{default_tokens_};
    int total_servers_{0};

    // array of seq. functions ids
    std::vector<int> seq_func_ids_;

    // array of new/old token functions
    std::vector<int> new_token_func_;

    // port service
    std::shared_ptr<zmq_port_service> port_service_;
    // data service
    std::shared_ptr<zmq_data_service> data_service_;

    //mutual exclusion data
    std::atomic_flag task_gen = ATOMIC_FLAG_INIT;
    
    // zeroMQ data
    zmq::context_t context_;
    std::unique_ptr<zmq::socket_t> replySock_;
    std::map<std::thread::id, std::shared_ptr<zmq::socket_t>> requestSockList_;
    //mutual exclusion data for socket map structure
    std::atomic_flag accessSockMap = ATOMIC_FLAG_INIT;

   
    /// server address
    std::thread server_thread_;
    
    // pool of threads
    int concurrency_degree_;
    mutable dist_pool<zmq_scheduler<task_type>> thread_pool_;

    // pointer to zmq_scheduler_thread
    zmq_scheduler_thread<Task> * zmq_sched_thread_ = NULL;

  /**
  \brief Function to create a zmq request socket for the port service
  \return Shared pointer with the zmq socket.
  */
  std::shared_ptr<zmq::socket_t> create_socket ()
  {
    COUT << "zmq_scheduler::create_socket begin\n";
   
    // create request socket shared pointer
    std::shared_ptr<zmq::socket_t> requestSock_ = std::make_shared<zmq::socket_t>(context_,ZMQ_REQ);

    // connect request socket
    std::ostringstream ss;
    ss << tcpConnectPattern[0] << machines_[server_id_] << tcpConnectPattern[1] << schedServer_port_;
    requestSock_->connect(ss.str());

    {std::ostringstream ss;
    ss  << "zmq_scheduler::create_socket connect: " << ss.str() << std::endl ;
    COUT << ss.str();}

    return requestSock_;
  }
};

template <typename Task>
class zmq_scheduler_thread{
  public:
    // Type alias for task type.
    using task_type = Task;
  
    // no copy constructors
    zmq_scheduler_thread(const zmq_scheduler_thread&) =delete;
    zmq_scheduler_thread& operator=(zmq_scheduler_thread const&) =delete;
    
    /**
    \brief Default construct a zmq scheduler thread
    */
    zmq_scheduler_thread(int maxTokens, int total_servers):
        seq_tasks_(),
        new_tok_par_tasks_(maxTokens),
        old_tok_par_tasks_(maxTokens),
        requests_(maxTokens),
        run_requests_(total_servers),
        end_requests_(total_servers),
        blocked_servers_(total_servers) {
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::constructor maxTokens = " << maxTokens << ", total_servers = " << total_servers << std::endl;
            COUT << ss.str();}
            maxTokens_ = maxTokens;
            total_servers_ = total_servers;
            new_tok_par_tasks_.registry(100);
        };

    /**
    \brief functor member
    */
    void operator() (zmq_scheduler<Task> * sched)
    {
        {std::ostringstream ss;
        ss << "zmq_scheduler_thread() maxTokens = " << maxTokens_ << ", total_servers_ = " << total_servers_ << std::endl;
        COUT << ss.str();}
        server_func (sched);
    }

  private:
  
    /// server data
    int tokens_{0};
    double ratio_create_tokens_{0.5};
    int running_threads_{0};
    int total_running_servers_{0};
    int total_ending_servers_{0};
    int total_blocked_servers_{0};
    // array of seq. functions ids
    std::vector<int> seq_func_ids_;
    // server id that executes the scheduler
    int sched_server_id_;
    // server ids for each server (server_id)
    std::vector<int> servers_ids_;
    // num. of threads for each server (server_id)
    std::vector<int> servers_num_threads_;
    // server for each sequential tasks (task.get_id(),server_id)
    std::map<int,int> seq_servers_;
    // sequential tasks queues (server_id, (task.get_id(),task))
    std::map<int,multi_queue<int,Task>> seq_tasks_;
    // new token parallel tasks queue (blk_server_id, task)
    multi_queue<int,Task> new_tok_par_tasks_;
    // old token parallel tasks queue (blk_server_id, task)
    multi_queue<int,Task> old_tok_par_tasks_;
    // waiting get requests queue (server_id, client_id)
    multi_queue<int,std::string> requests_;
    // queue for pending run request (client_id)
    locked_mpmc_queue<std::string> run_requests_;
    // queue for pending end request (client_id)
    locked_mpmc_queue<std::string> end_requests_;
    // queue for blocked servers (client_id,server_id,task)
    locked_mpmc_queue<std::tuple<std::string,int,Task>> blocked_servers_;
    // set of enabled sequential tasks (task.get_id())
    std::set<int> set_enabled_;
    // set of task that potentially can create new tokens (task.get_id())
    std::set<int> set_new_tok_tasks_;
    // set of task that cannot create new tokens (task.get_id())
    std::set<int> set_old_tok_tasks_;

    int maxTokens_{0};
    int total_servers_{0};
    
    unsigned int total_func_stages_{0};

    /**
    \brief Multiqueues initialization.
    
    Init thoses queues and structures that do not change
    between parallel executions of the same scheduler (Carea)
    
    \param machines map of <servers ids, machine IPs>
    */
    void queues_init (std::map<int, std::string> machines)
    {
      try {
        for (const auto& elem : machines) {
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::queues_init registry = " << elem.first << ", " << elem.second << std::endl;
            COUT << ss.str();}
            new_tok_par_tasks_.registry(elem.first);
            COUT << "zmq_scheduler_thread::queues_init new_tok_par_tasks_\n";

            old_tok_par_tasks_.registry(elem.first);
            COUT << "zmq_scheduler_thread::queues_init old_tok_par_tasks_\n";

            requests_.registry(elem.first);
            COUT << "zmq_scheduler_thread::queues_init requests_\n";

        }
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::queues_init: " << e.what() << std::endl;
      }
    }
    
    /**
    \brief Server initialization.
    */
    void server_init (zmq_scheduler<Task> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::server_init begin\n";

        // inproc server socket binded
        std::ostringstream ss;
        ss << sched->tcpBindPattern[0] << "0";
        sched->replySock_= std::make_unique<zmq::socket_t>(sched->context_,ZMQ_ROUTER);
        sched->replySock_->bind(ss.str());
        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::server_init bind: " << ss.str() << std::endl;
        COUT << ss.str();}

        size_t size = 256;
        char buf[256];
        COUT << "zmq_scheduler: getsockopt\n";
        sched->replySock_->getsockopt(ZMQ_LAST_ENDPOINT,buf,&size);
        std::string address(buf);
        std::string delimiter = ":";
        int pos = address.find(delimiter, address.find(delimiter)+1)+1;
        std::string srtPort = address.substr(pos); // token is "scott"
        {std::ostringstream ss;
        ss << "zmq_scheduler_thread: " << srtPort << std::endl;
        COUT << ss.str();}
        
        int port = atoi(srtPort.c_str());
        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::server_init " << address << " (" << sched->id_ << "," << port << ")\n";
        COUT << ss.str();}
        COUT << "zmq_scheduler_thread::server_init sched->port_service_->set begin\n";
        sched->port_service_->set(0,sched->schedServer_portserv_port_,port);
        COUT << "zmq_scheduler_thread::server_init sched->port_service_->set end\n";
      
        // init multiqueues
        queues_init (sched->machines_);
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::server_init: " << e.what() << std::endl;
      }
    }
    
    /**
    \brief Server function to store and release data form the storage array.
    */
    void server_func (zmq_scheduler<Task> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::server_func begin\n";

        // initialize server
        server_init (sched);
        COUT << "zmq_scheduler_thread::server_func:  server_init end\n";

        while (1) {

          zmq::message_t msg;
          std::string client_id;

          // receive client id
          COUT << "zmq_scheduler_thread::server_func sched->replySock_->recv begin\n";
          sched->replySock_->recv(&msg);
          //std::string client_id((char *)msg.data(), msg.size());
          client_id = std::string((char *)msg.data(), msg.size());
          COUT << "zmq_scheduler_thread::server_func sched->replySock_->recv client_id\n";

          // recv zero frame
          sched->replySock_->recv(&msg);
        
          if (msg.size() != 0) {
            std::string pru((const char* )msg.data(), (size_t) msg.size());
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::server_func ERROR frame zero: " << pru << "| size: " << msg.size() << std::endl;
            COUT << ss.str();}
            //throw std::runtime_error ("Error frame zero");
          }
          std::string pru((const char* )msg.data(), (size_t) msg.size());
          {std::ostringstream ss;
          ss << "zmq_scheduler_thread:: CMD: " << pru << "| size: " << msg.size() << std::endl;
          COUT << ss.str();}

          // recv command
          sched->replySock_->recv(&msg);
            
          // end task command
          if ( (msg.size() == sched->tskEndCmd.size()) &&
               (0 == std::memcmp(msg.data(),
                               static_cast<const void*>(sched->tskEndCmd.data()),
                               sched->tskEndCmd.size())) ) {
            // serving end command
            srv_end_cmd (sched, client_id);
        
          // run task command
          } else if ( (msg.size() == sched->tskRunCmd.size()) &&
                      (0 == std::memcmp(msg.data(),
                                      static_cast<const void*>(sched->tskRunCmd.data()),
                                      sched->tskRunCmd.size())) ) {
            // serving run command
            srv_run_cmd (sched, client_id);

          //  get task command
          } else if ( (msg.size() == sched->getTaskCmd.size()) &&
                      (0 == std::memcmp(msg.data(),
                                      static_cast<const void*>(sched->getTaskCmd.data()),
                                      sched->getTaskCmd.size())) ) {
            // serving get command
            srv_get_cmd (sched, client_id);

          // set task command
          } else if ( (msg.size() == sched->setTaskCmd.size()) &&
                      (0 == std::memcmp(msg.data(),
                                      static_cast<const void*>(sched->setTaskCmd.data()),
                                      sched->setTaskCmd.size())) ) {
            // serving set command
            bool status = srv_set_cmd (sched, client_id);
            if (status) continue;

          // consume task command
          } else if ( (msg.size() == sched->consumeCmd.size()) &&
                      (0 == std::memcmp(msg.data(),
                                      static_cast<const void*>(sched->consumeCmd.data()),
                                      sched->consumeCmd.size())) ) {
            // serving consume command
            srv_consume_cmd (sched, client_id);
          
          // ERROR
          } else {
            COUT << "zmq_scheduler_thread::server_func ERROR: Cmd not found\n";

          }
       
          //
          // server pending requests
          //
        
          // if there are pending requests -> try to assign them to pending tasks
          {std::ostringstream ss;
          ss << "zmq_scheduler_thread::server_func requests_.empty(): " << requests_.empty() << std::endl;
          COUT << ss.str();}

          if (! requests_.empty()) {

            // if no running tokens -> Check if we need to end the run,
            //                         the whole server or do nothing.
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::server_func tokens_: " << tokens_ << std::endl;
            COUT << ss.str();}

            if (tokens_ <= 0) {
          
              // check run and end received messages
              bool status = finish_execution (sched);
              //if everything is done, finish the thread
              if (status) break;
            
            } else {
              // get which servers has pending requests
              auto set_req_servers = requests_.loaded_set();
              {std::ostringstream ss;
              COUT << "zmq_scheduler_thread::server_func: set_req_servers.size(): " << set_req_servers.size() << std::endl;
              COUT << ss.str();}

              // check sequential tasks for all servers with pending requests
              exec_seq_task (sched, set_req_servers);
              // check parallel tasks with data on the same server for all remaining requests
              exec_par_task_same_server (sched, set_req_servers);
              // check remaining parallel tasks for all remaining requests
              exec_par_task_diff_server (sched);
            }
          }
        // no request loop again
        }
        COUT << "zmq_scheduler_thread::server_func end\n";
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::server_func: general exec: " << e.what() << std::endl;
     }

    }
 
 
    /**
    \brief End command service
    \param sched poiner to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_end_cmd (zmq_scheduler<Task> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::end_cmd_srv BEGIN\n";

        // receive server id
        zmq::message_t msg;
        sched->replySock_->recv(&msg);
        //int server_id = *((int*) msg.data());

        // store end request to wake it up at the end, increase ending servers
        end_requests_.push(client_id);
        total_ending_servers_++;
        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::srv_end_cmd: END total_ending_servers_=" << total_ending_servers_ << std::endl;
        COUT << ss.str();}
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_end_cmd" << e.what() << std::endl;
      }
    }
    
    /**
    \brief run command service
    \param sched poiner to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_run_cmd (zmq_scheduler<Task> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_run_cmd BEGIN\n";

        // recv number of threads
        zmq::message_t msg;
        sched->replySock_->recv(&msg);
        int num_threads = *((int*) msg.data());
        servers_num_threads_.push_back(num_threads);
    
        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::srv_run_cmd num_threads = " << num_threads << std::endl;
        COUT << ss.str();}

        // receive server id
        sched->replySock_->recv(&msg);
        int server_id = *((int*) msg.data());
        servers_ids_.push_back(server_id);

        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::srv_run_cmd server_id = " << server_id << std::endl;
        COUT << ss.str();}

        // receive if the server is the scheduler server
        sched->replySock_->recv(&msg);
        bool is_sched_server = ((*((int*) msg.data())) > 0);

        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::srv_run_cmd is_sched_server = " << is_sched_server << std::endl;
        COUT << ss.str();}

        if (is_sched_server) {
          // set it as the sched server
          sched_server_id_=server_id;
          // get list of seq. functions ids
          sched->replySock_->recv(&msg);
          seq_func_ids_.resize(msg.size()/sizeof(int));
          std::memcpy(seq_func_ids_.data(), msg.data(), msg.size());
           
          {std::ostringstream ss;
          ss << "zmq_scheduler_thread::srv_run_cmd seq_func_ids_.size() = " << seq_func_ids_.size() << std::endl;
          COUT << ss.str();}

          // get list of new token functions
          sched->replySock_->recv(&msg);
          total_func_stages_ = msg.size()/sizeof(int);
          std::vector<int> new_token_func_(total_func_stages_);
          std::memcpy(new_token_func_.data(), msg.data(), msg.size());
            
          {std::ostringstream ss;
          ss << "zmq_scheduler_thread::srv_run_cmd new_token_func_.size() = " << new_token_func_.size() << std::endl;
          COUT << ss.str();}

            // set new and ols tokens masks
          for (unsigned int i=0; i<new_token_func_.size(); i++) {
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::srv_run_cmd new_token_func_[" << i << "] = " << new_token_func_[i] << std::endl;
            COUT << ss.str();}

            if (new_token_func_[i] == 1) {
              set_new_tok_tasks_.insert(i);
            } else {
              set_old_tok_tasks_.insert(i);
            }
          }
          {std::ostringstream ss;
          ss << "zmq_scheduler_thread::srv_run_cmd:  set_new_tok_tasks_.size() = " << set_new_tok_tasks_.size() << ", set_old_tok_tasks_.size() = " << set_old_tok_tasks_.size() << std::endl;
          COUT << ss.str();}
          for (const auto &aux : set_new_tok_tasks_)  {
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::srv_run_cmd set_new_tok_tasks_  elem = " << aux << std::endl;
            COUT << ss.str();}
          }
          for (const auto &aux : set_old_tok_tasks_)  {
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::srv_run_cmd set_old_tok_tasks_  elem = " << aux << std::endl;
            COUT << ss.str();}
          }
        }

        // store run request to wake it up at the end, increase running servers and threads
        run_requests_.push(client_id);
        total_running_servers_++;
        running_threads_= running_threads_ + num_threads;

        COUT << "zmq_scheduler_thread::srv_run_cmd total_running_servers_ = " << total_running_servers_ << std::endl;
        COUT << "zmq_scheduler_thread::srv_run_cmd sched->total_servers_ = " << sched->total_servers_ << std::endl;

        // if all run requests have arrived, launch initial task
        // NOTE: inital task is supposed to create new tokens???
        if (total_running_servers_ == sched->total_servers_) {
        
          // fill up asignation of sequential tasks to servers.
          // also enable sequential tasks.
          unsigned int indx = 0;
          for (unsigned int i=0; i<seq_func_ids_.size(); i++) {
            seq_servers_[seq_func_ids_[i]]=servers_ids_[indx];
            set_enabled_.insert(seq_func_ids_[i]);
            indx++;
            if (indx >=servers_ids_.size()) indx=0;
          }
          COUT << "zmq_scheduler_thread::srv_run_cmd set_enabled_.size() = " << set_enabled_.size() << std::endl;
          for (const auto &aux : set_enabled_)  {
            COUT << "zmq_scheduler_thread::srv_run_cmd set_enabled_  elem = " << aux << std::endl;
          }
          // create all the multiqueues of the seq map
          for (unsigned int i=0; i<servers_ids_.size(); i++) {
            seq_tasks_.emplace(std::piecewise_construct,
                               std::forward_as_tuple(servers_ids_[i]),
                               std::forward_as_tuple(sched->max_tokens_));
            for (unsigned int j=0; j<total_func_stages_; j++) {
              seq_tasks_.find(servers_ids_[i])->second.registry(j);
            }
          }
          COUT << "zmq_scheduler_thread::srv_run_cmd servers_ids_.size() = " << servers_ids_.size() << std::endl;
            
          if (seq_servers_.find(0) != seq_servers_.end()) {
            //run initial seq task (0) on its corresp. server
            seq_tasks_.find(seq_servers_[0])->second.push(0, Task{0,0});
            COUT << "zmq_scheduler_thread::srv_run_cmd INIT SEQ = \n";
          } else {
            // run initial task on first server
            new_tok_par_tasks_.push(servers_ids_[0],Task{0,0});
            COUT << "zmq_scheduler_thread::srv_run_cmd INIT PAR = \n";
          }
          assert(tokens_==0);
          tokens_=1;
          COUT << "zmq_scheduler_thread::end_cmd_srv END\n";
        }
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_run_cmd: " << e.what() << std::endl;
      }
    }
    
    /**
    \brief get command service
    \param sched poiner to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_get_cmd (zmq_scheduler<Task> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_get_cmd  BEGIN\n";
        
        // recv server_id and task_id
        zmq::message_t msg;
        sched->replySock_->recv(&msg);
        int server_id = *((int*) msg.data());
        sched->replySock_->recv(&msg);
        Task old_task{};
        old_task.set_serialized_string((char *)msg.data(),msg.size());

        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::srv_get_cmd: server_id: "
                  << server_id << ", old_task: (" << old_task.get_id()
                  << ", " << old_task.get_task_id() << ")\n";
        COUT << ss.str();}
        // if a sequential task ended, enable it
        if ( (old_task.get_id() >= 0) &&
             (seq_servers_.find(old_task.get_id()) != seq_servers_.end() ) ) {
          set_enabled_.insert(old_task.get_id());
          COUT << "zmq_scheduler_thread::srv_get_cmd: seq task ended\n";
        }

        // push task request to be served latter
        requests_.push(server_id,client_id);
        COUT << "zmq_scheduler_thread::srv_get_cmd END\n";
          
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_get_cmd: " << e.what() << std::endl;
      }
    }

    /**
    \brief set command service
    \param sched poiner to the scheduler object
    \param client_id zmq client identificator for answering command.
    \return booolean value true -> break execution; false -> continue normal execution.
    */
    bool srv_set_cmd (zmq_scheduler<Task> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_set_cmd  BEGIN\n";
          
        // recv server_id, task and new_token
        zmq::message_t msg;
        sched->replySock_->recv(&msg);
        COUT << "zmq_scheduler_thread::srv_set_cmd  recv_ B\n";
        int server_id = *((int*) msg.data());
        sched->replySock_->recv(&msg);
        COUT << "zmq_scheduler_thread::srv_set_cmd  recv_ C\n";
        Task task{};
        task.set_serialized_string((char *)msg.data(),msg.size());
        sched->replySock_->recv(&msg);
        COUT << "zmq_scheduler_thread::srv_set_cmd  recv_ D\n";
        bool new_token = ( *((int*) msg.data()) > 0 );

        {std::ostringstream ss;
        ss << "zmq_scheduler_thread::srv_set_cmd: server_id: "
                  << server_id << ", task: (" << task.get_id()
                  << ", " << task.get_task_id() << ")\n";
        COUT << ss.str();}
        
        // if task requires a new token check if possible and then increase them
        if (new_token) {
          if (tokens_ >= sched->max_tokens_) {
            // if tokens can't be increased block the server until they can
            blocked_servers_.push(make_tuple(client_id,server_id,task));
            total_blocked_servers_++;

            COUT << "zmq_scheduler_thread::srv_set_cmd: total_blocked_servers_" << total_blocked_servers_ << std::endl;

            if (total_blocked_servers_ >= sched->total_servers_) {
              // if all servers are blocked, launch a run-time exception.
              throw std::runtime_error ("All servers blocked");
            }
            // stop and jump to read next request
            COUT << "zmq_scheduler_thread::srv_set_cmd: continue"  << std::endl;
            return (true);
            
          }
          tokens_++;
          COUT << "zmq_scheduler_thread::srv_set_cmd: tokens_" << tokens_ << std::endl;

        }
        
        COUT << "zmq_scheduler_thread::srv_set_cmd: check task\n";

        // if new task is sequential, insert it on the corresponding server's queue
        if (seq_servers_.find(task.get_id()) != seq_servers_.end()) {
          seq_tasks_.find(seq_servers_[task.get_id()])->second.push(task.get_id(), task);
          COUT << "zmq_scheduler_thread::srv_set_cmd: NEW TASK SEQ\n";

        // if new task could create new tokens, insert on new_token_parallel_queue
        } else if (set_new_tok_tasks_.find(task.get_id())!=set_new_tok_tasks_.end()) {
          new_tok_par_tasks_.push(server_id,task);
          COUT << "zmq_scheduler_thread::srv_set_cmd: NEW TASK NEW_TOKEN\n";
        // else insert on old_token_parallel_queue
        } else {
          old_tok_par_tasks_.push(server_id,task);
          COUT << "zmq_scheduler_thread::srv_set_cmd: NEW TASK OLD_TOKEN\n";
        }

        // send back an ACK
        sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
        sched->replySock_->send("", 0, ZMQ_SNDMORE);
        sched->replySock_->send("", 0);
       
        COUT << "zmq_scheduler_thread::srv_set_cmd END\n";
        
       } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_set_cmd: " << e.what() << std::endl;
      }
      return (false);
    }

    /**
    \brief consume command service
    \param sched poiner to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_consume_cmd (zmq_scheduler<Task> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_consume_cmd  BEGIN\n";
          
        // consume one token and check if there is blocked servers
        if (! blocked_servers_.empty()) {
          // insert new task from blocked server and wake it up
          auto data = blocked_servers_.pop();
          auto blk_client_id = std::get<0>(data);
          auto blk_server_id = std::get<1>(data);
          auto blk_task = std::get<2>(data);
          total_blocked_servers_--;

          COUT << "zmq_scheduler_thread::srv_consume_cmd: total_blocked_servers_" << total_blocked_servers_ << std::endl;

          // if new task is sequential, insert it on the corresponding server's queue
          if (seq_servers_.find(blk_task.get_id()) != seq_servers_.end()) {
            seq_tasks_.find(seq_servers_[blk_task.get_id()])->
                            second.push(blk_task.get_id(), blk_task);
              COUT << "zmq_scheduler_thread::srv_consume_cmd: NEW TASK SEQ\n";

          // if new task could create new tokens, insert on new_token_parallel_queue
          } else if (set_new_tok_tasks_.find(blk_task.get_id()) != set_new_tok_tasks_.end()) {
            new_tok_par_tasks_.push(blk_server_id,blk_task);
            COUT << "zmq_scheduler_thread::srv_consume_cmd: NEW TASK NEW_TOKEN\n";

          // else insert on old_token_parallel_queue
          } else {
            old_tok_par_tasks_.push(blk_server_id,blk_task);
            COUT << "zmq_scheduler_thread::srv_consume_cmd: NEW TASK OLD_TOKEN\n";

          }
          // send back an ACK to the blocked server
          sched->replySock_->send(blk_client_id.data(),
                                 blk_client_id.size(),
                                 ZMQ_SNDMORE);
          sched->replySock_->send("", 0, ZMQ_SNDMORE);
          sched->replySock_->send("", 0);
          COUT << "zmq_scheduler_thread::srv_consume_cmd: ACK SENT for SET TASK\n";

        } else {
          tokens_--;
          COUT << "zmq_scheduler_thread::srv_consume_cmd: tokens_" << tokens_ << std::endl;

        }
        
        // send back an ACK for consume request
        sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
        sched->replySock_->send("", 0, ZMQ_SNDMORE);
        sched->replySock_->send("", 0);
        COUT << "zmq_scheduler_thread::srv_consume_cmd: ACK SENT \n";

       } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_consume_cmd: " << e.what() << std::endl;
      }
    }

    /**
    \brief Finalize execution (treats run and end messages received)
    \param sched pointer to the scheduler object
    \return booolean value true -> Finish server thread; false -> continue normal execution.

    */
    bool finish_execution (zmq_scheduler<Task> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::finish_execution  BEGIN\n";

        // if run request are all received then finish them
        // note: run request can not be all recieved with no tokens
        //       unless is the end of a run
        //       (when last run request arrive, tokens is set to 1 immediately)
        COUT << "zmq_scheduler_thread::finish_execution total_running_servers_: " << total_running_servers_ << ", sched->total_servers_: " << sched->total_servers_ << std::endl;
        if (total_running_servers_ == sched->total_servers_) {
          while (! run_requests_.empty()) {
            // get run reply address and send back and ACK
            auto client_id = run_requests_.pop();
            sched->replySock_->send(client_id.data(),
                                    client_id.size(), ZMQ_SNDMORE);
            sched->replySock_->send("", 0, ZMQ_SNDMORE);
            sched->replySock_->send("", 0);
            COUT << "zmq_scheduler_thread::finish_execution SENT RUN ACK "<< std::endl;
          }

          // reset counter of running servers
          total_running_servers_ = 0;
          // reset counter of running threads
          running_threads_ = 0;
          //reset all set of flags
          set_enabled_.clear();
          set_new_tok_tasks_.clear();
          set_old_tok_tasks_.clear();
          // clear seq_servers_ and seq_tasks_ maps
          seq_servers_.clear();
          seq_tasks_.clear();
          // erase servers_num_threads_ servers_ids_ and  seq_func_ids_ vectors
          servers_num_threads_.clear();
          servers_ids_.clear();
          seq_func_ids_.clear();
          // set sched_server_id_ to 0
          sched_server_id_=0;
        }
        // if all end request are received (after finishing run requests)
        // finish set task request to kill working threads,
        // finish end threads, and end sched thread
        COUT << "zmq_scheduler_thread::finish_execution total_running_servers_: " << total_running_servers_ << ", total_ending_servers_: " << total_ending_servers_ << ", sched->total_servers_: " << sched->total_servers_ << std::endl;
        if ( (total_running_servers_ == 0)  &&
             (total_ending_servers_ == sched->total_servers_) ) {
          // finish get task request
          // send a null task to all pending requests
          while (! requests_.empty()) {
            auto client_id = requests_.pop();
            sched->replySock_->send(client_id.data(),
                                    client_id.size(), ZMQ_SNDMORE);
            sched->replySock_->send("", 0, ZMQ_SNDMORE);
            Task task; // default/empty task
            std::string task_string = task.get_serialized_string();
            sched->replySock_->send((char *)(task_string.data()), task_string.size());
              
            // decrease running threads.
            // If all done wake up all end replies and end sched. thread
            running_threads_--;
            {std::ostringstream ss;
            ss << "zmq_scheduler_thread::finish_execution: SENT GET TASK ACK NULL: task: (" << task.get_id() << ", " << task.get_task_id() << ")\n";
            COUT << ss.str();}

          }
          COUT << "zmq_scheduler_thread::finish_execution: running_threads_ = "<< running_threads_ << std::endl;

          if (running_threads_ <= 0) {
            while (! end_requests_.empty()) {
              // get run reply address and send back and ACK
              auto client_id = end_requests_.pop();
              sched->replySock_->send(client_id.data(),
                                      client_id.size(), ZMQ_SNDMORE);
              sched->replySock_->send("", 0, ZMQ_SNDMORE);
              sched->replySock_->send("", 0);
              COUT << "zmq_scheduler_thread::finish_execution SENT END ACK \n";

            }
            // reset counter of ending servers
            total_ending_servers_ = 0;
                  
            //
            // end Scheduling thread
            //
            return (true);
          }
        }
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::finish_execution: " << e.what() << std::endl;
      }
      return (false);
    }

    /**
    \brief Check sequential tasks for all servers with pending requests.
    */
    void exec_seq_task (zmq_scheduler<Task> * sched,
                        std::set<int> set_req_servers)
    {
      try {
        static int index_new = 0;
        static int index_old = 0;
      
        COUT << "zmq_scheduler_thread::exec_seq_task BEGIN\n";

        int index = 0;
        auto it=set_req_servers.begin();
        while (it!=set_req_servers.end()) {
        
          int search_end = false;
          do {

            auto set_seq_task_new =
               seq_tasks_.find(*it)->second.loaded_set(set_enabled_,set_new_tok_tasks_);
            auto set_seq_task_old =
               seq_tasks_.find(*it)->second.loaded_set(set_enabled_,set_old_tok_tasks_);

            COUT << "zmq_scheduler_thread::exec_seq_task: check new seq server with request: set_seq_task_new.size() = " << set_seq_task_new.size() << ", set_seq_task_old.size() = " << set_seq_task_old.size() << std::endl;

            double actual_ratio = ((double)tokens_) / ((double) sched->max_tokens_);
            COUT << "zmq_scheduler_thread::exec_seq_task: actual_ratio = " << actual_ratio << ", ratio_create_tokens_ = " << ratio_create_tokens_ << std::endl;

            // if tokens ratio is below the threshold and there are new token tasks, select one
            if ( (actual_ratio < ratio_create_tokens_) &&
                 (! set_seq_task_new.empty()) ) {
            
              COUT << "zmq_scheduler_thread::exec_seq_task: get seq new token\n";

              // check next task counting from last index_new, select one and update index_new
              auto it2 = set_seq_task_new.lower_bound(index_new);
              if (it2 == set_seq_task_new.end()) {
                it2=set_seq_task_new.begin();
              }
              index = *it2;
              index_new = index;
                
              // erase selected index from current enable and new tokens sets
              set_enabled_.erase(index);
              //set_new_tok_tasks_.erase(index);
                
              // else if there are old token tasks, select one
            } else if (! set_seq_task_old.empty()) {
              
              COUT << "zmq_scheduler_thread::exec_seq_task: get seq old token\n";

              // check next task counting from last index_old, select one and update index_old
              auto it2 = set_seq_task_old.lower_bound(index_old);
              if (it2 == set_seq_task_old.end()) {
                it2=set_seq_task_old.begin();
              }
              index = *it2;
              index_old = index;

              // erase selected index from current enable and old tokens sets
              set_enabled_.erase(index);
              //set_old_tok_tasks_.erase(index);
            
            // else end checking seq. tasks for this server
            } else {
              it++;
              search_end = true;
              COUT << "zmq_scheduler_thread::exec_seq_task: A- search_end = TRUE\n";

            }
            
            // if an index have been found
            if (! search_end) {
              // get the task, the request and reply it
              auto task = seq_tasks_.find(*it)->second.pop(index);
              auto client_id = requests_.pop(*it);
            
              COUT   << "zmq_scheduler_thread::exec_seq_task: id_: "
                        << *it << ", task: (" << task.get_id()
                        << ", " << task.get_task_id() << ")\n";

              sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
              sched->replySock_->send("", 0, ZMQ_SNDMORE);
              std::string task_string = task.get_serialized_string();
              sched->replySock_->send((char *)(task_string.data()), task_string.size());

              {std::ostringstream ss;
              ss << "zmq_scheduler_thread::exec_seq_task: SENT GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")\n";
              COUT << ss.str();}
              
              // if no more requests for this server, erase server from set and end the search
              if (requests_.empty(*it)) {
                it = set_req_servers.erase(it);
                search_end = true;
                COUT << "zmq_scheduler_thread::exec_seq_task: B- search_end = TRUE\n";

              }
            }
          } while (! search_end);
        }
      COUT << "zmq_scheduler_thread::exec_seq_task END\n";
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::exec_seq_task: " << e.what() << std::endl;
      }
    }
  
    /**
    \brief Check parallel tasks with data on the same server for all remaining requests
    */
    void exec_par_task_same_server (zmq_scheduler<Task> * sched,
                                    std::set<int> set_req_servers)
    {
      try {
        COUT << "zmq_scheduler_thread::exec_par_task_same_server BEGIN\n";

        auto it=set_req_servers.begin();
        while (it!=set_req_servers.end()) {
          COUT << "zmq_scheduler_thread::exec_par_task_same_server: check new server with request\n";

          int search_end = false;
          do {
            Task task;
            double actual_ratio = ((double)tokens_) / ((double) sched->max_tokens_);
          
            COUT << "zmq_scheduler_thread::exec_par_task_same_server: actual_ratio = " << actual_ratio << ", ratio_create_tokens_ = " << ratio_create_tokens_ << std::endl;

            // if there are new token parallel tasks, select one
            if ( (actual_ratio < ratio_create_tokens_) &&
                 (! new_tok_par_tasks_.empty(*it)) ) {
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: server new token req\n";

              task = new_tok_par_tasks_.pop(*it);
            
            // else if there are old token parallel tasks, select one
            } else if (! old_tok_par_tasks_.empty(*it)) {
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: server old token req\n";
              task = old_tok_par_tasks_.pop(*it);
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: server old token req END\n";

            // else end checking parallel tasks for this server
            } else {
              it++;
              search_end = true;
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: A- search_end = TRUE\n";

            }
            
            // if a task have been found
            if (! search_end) {
              {std::ostringstream ss;
              ss << "zmq_scheduler_thread::exec_par_task_same_server: START SENDING GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")\n";
              COUT << ss.str();}

              // get the task, the request and reply it
              auto client_id = requests_.pop(*it);
              sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
              sched->replySock_->send("", 0, ZMQ_SNDMORE);
              std::string task_string = task.get_serialized_string();
              sched->replySock_->send((char *)(task_string.data()), task_string.size());

              {std::ostringstream ss;
              ss << "zmq_scheduler_thread::exec_par_task_same_server: SENT GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")\n";
              COUT << ss.str();}


              // if no more requests for this server, erase server from set and end the search
              if (requests_.empty(*it)) {
                it = set_req_servers.erase(it);
                search_end = true;
                COUT << "zmq_scheduler_thread::exec_par_task_same_server: B- search_end = TRUE\n";

              }
            }
          } while (! search_end);
        }
        COUT << "zmq_scheduler_thread::exec_par_task_same_server END\n";

      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::exec_par_task_same_server: " << e.what() << std::endl;
      }
    }

  
    /**
    \brief Check remaining parallel tasks for all remaining requests
    */
    void exec_par_task_diff_server (zmq_scheduler<Task> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::exec_par_task_diff_server BEGIN\n";
 
        while (! requests_.empty()) {
          COUT << "zmq_scheduler_thread::exec_par_task_diff_server: check new request\n";
          Task task;
          double actual_ratio = ((double)tokens_) / ((double) sched->max_tokens_);
          // if there are new token parallel tasks, select one
          COUT << "zmq_scheduler_thread::exec_par_task_diff_server: actual_ratio = " << actual_ratio << ", ratio_create_tokens_ = " << ratio_create_tokens_ << std::endl;
          if ( (actual_ratio < ratio_create_tokens_) &&
               (! new_tok_par_tasks_.empty()) ) {
            COUT << "zmq_scheduler_thread::exec_par_task_diff_server: server old token req\n";
            task = new_tok_par_tasks_.pop();
        
            // else if there are old token parallel tasks, select one
          } else if (! old_tok_par_tasks_.empty()) {
            COUT << "zmq_scheduler_thread::exec_par_task_diff_server: server old token req\n";
            task = old_tok_par_tasks_.pop();
            
          // else end checking parallel tasks
          } else {
            COUT << "zmq_scheduler_thread::exec_par_task_diff_server: No more request served\n";

            break;
          }
        
          // get the task, the request and reply it
          auto client_id = requests_.pop();
          sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
          sched->replySock_->send("", 0, ZMQ_SNDMORE);
          std::string task_string = task.get_serialized_string();
          sched->replySock_->send((char *)(task_string.data()), task_string.size());

          {std::ostringstream ss;
          ss << "zmq_scheduler_thread::exec_par_task_diff_server: SENT GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")\n";
          COUT << ss.str();}
        }
        COUT << "zmq_scheduler_thread::exec_par_task_diff_server END\n";
      
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::exec_par_task_diff_server: " << e.what() << std::endl;
      }
    }
};


template <typename Task>
void zmq_scheduler<Task>::launch_thread() {
    zmq_sched_thread_ = new zmq_scheduler_thread<Task>{max_tokens_, total_servers_};
    server_thread_ = std::thread(std::ref(*zmq_sched_thread_),this);
}

}
#endif
