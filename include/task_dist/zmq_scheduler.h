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
#include <iterator>
#include <assert.h>

#include <zmq.hpp>

#include "../common/mpmc_queue.h"
#include "../common/configuration.h"

#include "dist_pool.h"
#include "zmq_data_service.h"
#include "zmq_port_service.h"
#include "multi_queue.h"
#include "multi_queue_hard.h"

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

#pragma GCC diagnostic warning "-Wunused-parameter"

template <typename Task>
class zmq_scheduler_thread;

template <typename Task>
class zmq_scheduler{
  public:
    // Type alias for task type and data reference.
    using task_type = Task;
    using data_ref_type = typename zmq_data_service::data_ref_type;
    static_assert(std::is_same<data_ref_type,typename task_type::data_ref_type>::value);

    // no copy constructors
    zmq_scheduler(const zmq_scheduler&) =delete;
    zmq_scheduler& operator=(zmq_scheduler const&) =delete;

    void launch_thread();

    /**
    \brief Default construct a zmq scheduler
    */
    zmq_scheduler(std::map<long, std::string> machines, long id,
              const std::shared_ptr<zmq_port_service> &port_service,
              long numTokens, long server_id, long concurrency_degree= configuration<>{}.concurrency_degree()) :
      machines_{machines.begin(), machines.end()},
      id_{id},
      server_id_{server_id},
      is_sched_server_{id == server_id},
      schedServer_portserv_port_{port_service->new_port()},
      max_tokens_{numTokens},
      total_servers_{(long)machines_.size()},
      port_service_{port_service},
      context_{1},
      concurrency_degree_{concurrency_degree}
    {
      // get total machines and the order of local machine in machines_ map
      total_machines_ = machines_.size();
      machine_map_order_ = std::distance(std::begin(machines_), machines_.find(id_)) + 1;
      COUT << "zmq_scheduler::zmq_scheduler total_machines_=" << total_machines_
         << ", machine_map_order_=" <<  machine_map_order_ << ENDL;

      functions.reserve(max_functions);
      COUT << "zmq_scheduler::zmq_scheduler data_service_" << ENDL;;
      data_service_ = std::make_shared<zmq_data_service>(machines_, id_,
                                                         port_service_, max_tokens_);
      COUT << "zmq_scheduler::zmq_scheduler data_service_ done" << ENDL;

      // if server, bind reply socket and launch thread
      if (is_sched_server_) {
        // server thread launched
        COUT << "zmq_scheduler::zmq_scheduler launch_thread()" << ENDL;
        launch_thread();
        COUT << "zmq_scheduler::zmq_scheduler launch_thread() done" << ENDL;
      }

      // get secheduler server port
      COUT << "zmq_scheduler::zmq_scheduler port_service_->get " << ENDL;
      schedServer_port_ = port_service_->get(0,schedServer_portserv_port_, true);
      COUT << "zmq_scheduler::zmq_scheduler port_service_->get end " << ENDL;
   
      // launch thread pool
      thread_pool_.init(this,concurrency_degree_);

    };

    /**
    \brief Default destrucutor for a zmq scheduler
    */
    ~zmq_scheduler() {
      COUT << "zmq_scheduler::~zmq_scheduler BEGIN" << ENDL;
      end();
      thread_pool_.finalize_pool();
      if (is_sched_server_) {
        server_thread_.join();
      }
      COUT << "zmq_scheduler::~zmq_scheduler END" << ENDL;
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
    long register_sequential_task(std::function<void(task_type&)> && f,
                                bool create_tokens)
    {
      while(task_gen_.test_and_set());
      long function_id = functions.size();;
      functions.emplace_back(f);
      seq_func_ids_.push_back(function_id);
      if (create_tokens) {
        new_token_func_.push_back(1);
      } else {
        new_token_func_.push_back(0);
      }
      task_gen_.clear();
      COUT << "register_sequential_task: func_id=" << function_id << ENDL;
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
   long register_parallel_task(std::function<void(task_type&)> && f,
                                bool create_tokens)
   {
     while(task_gen_.test_and_set());
     long function_id = (long) functions.size();
     functions.emplace_back(f);
     if (create_tokens) {
       new_token_func_.push_back(1);
     } else {
       new_token_func_.push_back(0);
     }
     task_gen_.clear();
     COUT << "register_parallel_stage: func_id=" << function_id << ENDL;
     return function_id;
   }

   /**
   \brief Removes the functions from the function lists.
   Remove the registered functions and their control variables.
   */
   void clear_tasks()
   {
     while(task_gen_.test_and_set());
    
     // reset functions vector
     functions.clear();
     functions.reserve(max_functions);

     // reset functions vector
     seq_func_ids_.clear();
     seq_func_ids_.reserve(max_functions);
     task_gen_.clear();
   }
   //**********   Managing  task ids
   /**
   \brief Get a new global task id based on the local task id and the machines_ distance.
   Get a new global task id based on the local task id and the machines_ distance.
   \return new global task id
   */


   long get_task_id()
   {
     while(task_ids_gen_.test_and_set());
     // compute global task id
     long task_id = (last_local_task_id_ * total_machines_) + machine_map_order_;
     // increase local task id
     last_local_task_id_++;
     task_ids_gen_.clear();
     
     COUT << "zmq_scheduler::get_task_id task_id=" << task_id << ENDL;
     return task_id;
   }
   
   //**********   getting node ids
   /**
   \brief Get the node id which is the index in the machines_ list.
   \return node id
   */


   long get_node_id()
   {
     return id_;
   }

   //**********   client part of the server messages tasks
   
   /**
   \brief Introduces a task into the pending tasks queue.

   Introduces a task from a stream pattern in the pending task queue. 
   \param task new task to be launched.
   \param new_token new task needs a new token or not.
   */
   void set_task(const task_type & task, bool new_token)
   {
     try {
       COUT << "zmq_scheduler:set_task BEGIN" << ENDL;

       // Get the socket for this thread
       while(accessSockMap_.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap_.clear();
       COUT << "zmq_scheduler:set_task requestSock_ obtain" << ENDL;

       long new_token_int = (new_token ? 1 : 0);
       COUT << "zmq_scheduler:set_task requestSock_ A" << ENDL;
       requestSock_->send(setTaskCmd.data(), setTaskCmd.size(), ZMQ_SNDMORE);
       COUT << "zmq_scheduler:set_task requestSock_ B" << ENDL;
       requestSock_->send((long *)(&id_), sizeof(id_), ZMQ_SNDMORE);
       COUT << "zmq_scheduler:set_task requestSock_ C" << ENDL;
       task.send(*requestSock_,ZMQ_SNDMORE);
       COUT << "zmq_scheduler:set_task requestSock_ D" << ENDL;
       requestSock_->send((long *)(&new_token_int), sizeof(new_token_int));

       COUT << "zmq_scheduler::set_task SENT: id_: "
          << id_ << ", task: (" << task.get_id()
          << ", " << task.get_task_id()
          << "), new_token_int = " << new_token_int << ENDL;

       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       COUT << "zmq_scheduler::set_task: size=" << message.size() << ENDL;

       if (message.size() != 0) {
          COUT << "Error: Wrong ACK" << ENDL;
          throw std::runtime_error("ERROR: Wrong ACK");
       }
       COUT << "zmq_scheduler:set_task END" << ENDL;

     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:set_task" << e.what() << std::endl;;
       std::throw_with_nested( std::runtime_error("zmq_scheduler:set_task"));
     }
   }
   
   /**
   \brief Get a new task.

   Get a task in a blocking mode for those threads that has launched 
   new dependent tasks.
   \param old_task old task that have been executed.
   \return new task to be run
   */
   task_type get_task(const task_type & old_task)
   {
     try {
       COUT << "zmq_scheduler:get_task BEGIN" << ENDL;
       // Get the socket for this thread
       while(accessSockMap_.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap_.clear();
       COUT << "zmq_scheduler:get_task requestSock_ obtain" << ENDL;

       //auto old_task_id = old_task.get_task_id();
       requestSock_->send(getTaskCmd.data(), getTaskCmd.size(), ZMQ_SNDMORE);
       requestSock_->send((long *)(&id_), sizeof(id_), ZMQ_SNDMORE);
       old_task.send(*requestSock_);
       
       COUT << "zmq_scheduler::get_task SENT: id_: "
          << id_ << ", old_task: (" << old_task.get_id()
          << ", " << old_task.get_task_id() << ")" << ENDL;
       
       // receive the data
       task_type new_task{};
       new_task.recv(*requestSock_);

       COUT << "zmq_scheduler:get_task END " << ENDL;
       return (new_task);
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:get_task" << e.what() << std::endl;
       return (task_type{});
     }
   }

   /**
   \brief Notifies the ending of a full task (task_id).
   
   Notifies the ending of a full task (task_id).
   \param old_task old task that have been executed.
   \param num_tokens number of tokens to free (default=1).
   */
   void finish_task(const task_type & old_task, long num_tokens=1)
   {
     try {
       COUT << "zmq_scheduler:finish_task BEGIN " << ENDL;
       // Get the socket for this thread
       while(accessSockMap_.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap_.clear();
       COUT << "zmq_scheduler:finish_task requestSock_ obtain" << ENDL;

       requestSock_->send(finishTaskCmd.data(), finishTaskCmd.size(), ZMQ_SNDMORE);
       COUT << "zmq_scheduler:finish_task SENT" << ENDL;
       old_task.send(*requestSock_,ZMQ_SNDMORE);
       requestSock_->send((long *)(&num_tokens), sizeof(num_tokens));

       COUT << "zmq_scheduler::finish_task SENT: num_tokens: "
          << num_tokens << ", old_task: (" << old_task.get_id()
          << ", " << old_task.get_task_id() << ")" << ENDL;

       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       COUT << "zmq_scheduler::finish_task: size=" << message.size() << ENDL;

       if (message.size() != 0) {
          COUT << "Error: Wrong ACK" << ENDL;
          throw std::runtime_error("ERROR: Wrong ACK");
       }
       COUT << "zmq_scheduler: finish_task END" << ENDL;
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:finish_task" << e.what() << std::endl;
       std::throw_with_nested( std::runtime_error("zmq_scheduler:finish_task "));
     }
   }

   /**
   \brief starts a stream pattern form each machine.
  
   Starts a stream pattern form each machine. When all has notified the running
   command it will start, all run functions will block until all is done.
   \return last executed task on this run
   */
   task_type run()
   {
     try {
       COUT << "zmq_scheduler:run BEGIN" << ENDL;
       // Get the socket for this thread
       while(accessSockMap_.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap_.clear();
       COUT << "zmq_scheduler::run requestSock_ obtain" << ENDL;

       long is_sched_server_int = (is_sched_server_ ? 1 : 0);

       requestSock_->send(tskRunCmd.data(), tskRunCmd.size(), ZMQ_SNDMORE);
       requestSock_->send((long *)(&concurrency_degree_), sizeof(concurrency_degree_), ZMQ_SNDMORE);
       requestSock_->send((long *)(&id_), sizeof(id_), ZMQ_SNDMORE);
       if (is_sched_server_) {
           requestSock_->send((long *)(&is_sched_server_int), sizeof(is_sched_server_int), ZMQ_SNDMORE);
           requestSock_->send(seq_func_ids_.data(), seq_func_ids_.size() * sizeof(long), ZMQ_SNDMORE);
           requestSock_->send(new_token_func_.data(), new_token_func_.size() * sizeof(long));

           COUT << "zmq_scheduler::run SENT: concurrency_degree_: "
              << concurrency_degree_ << ", id_: " << id_
              << ", is_sched_server_" << is_sched_server_
              << ", seq_func_ids_.size()" << seq_func_ids_.size()
              << ", new_token_func_.size() = " << new_token_func_.size()
              << ENDL;
       } else {
           requestSock_->send((long *)(&is_sched_server_int), sizeof(is_sched_server_int));

           COUT << "zmq_scheduler::run SENT: concurrency_degree_: "
              << concurrency_degree_ << ", id_: " << id_
              << ", is_sched_server_" << is_sched_server_
              << ENDL;
       }

       // receive the data
       task_type new_task{};
       new_task.recv(*requestSock_);
       
       COUT << "zmq_scheduler::run last_task: (" << new_task.get_id()
          << ", " << new_task.get_task_id()
          << ")" << ENDL;

       
       // clean task arrays for next execution
       clear_tasks();
       COUT << "zmq_scheduler:run END" << ENDL;
       return new_task;
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:run" << e.what() << std::endl;
       return {};
     }
   }

   /**
   \brief ends the scheduler and scheduler thread.
   Sends a message to the scheduler thread to end and to end the working threads.
   */
   void end()
   {
     try {
       COUT << "zmq_scheduler:end BEGIN" << ENDL;
       // Get the socket for this thread
       while(accessSockMap_.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap_.clear();
       COUT << "zmq_scheduler::end requestSock_ obtain" << ENDL;

       requestSock_->send(tskEndCmd.data(), tskEndCmd.size(), ZMQ_SNDMORE);
       requestSock_->send((long *)(&id_), sizeof(id_));
       
       COUT << "zmq_scheduler::end SENT: id_: " << id_  << ENDL;
       
       // receive the data
       zmq::message_t message;
       requestSock_->recv(&message);
       COUT << "zmq_scheduler::end: size=" << message.size() << ENDL;

       if (message.size() != 0) {
          COUT << "Error: Wrong ACK" << ENDL;
          throw std::runtime_error("ERROR: Wrong ACK");
       }
       COUT << "zmq_scheduler:end END" << ENDL;
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:end" << e.what() << std::endl;
       std::throw_with_nested( std::runtime_error("zmq_scheduler:end "));
     }
   }
   
   /**
   \brief Pre-allocate a number of tokens if possible
   \param num_tokens number of tokens to allocate
   \return tokens allocated (true) or not (false)
   */
   bool allocate_tokens(long num_tokens)
   {
     try {
       COUT << "zmq_scheduler:allocate_tokens BEGIN" << ENDL;
       // Get the socket for this thread
       while(accessSockMap_.test_and_set());
       if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
          requestSockList_.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(std::this_thread::get_id()),
                                   std::forward_as_tuple(create_socket()));
       }
       std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
       accessSockMap_.clear();
       COUT << "zmq_scheduler::allocate_tokens requestSock_ obtain" << ENDL;

       requestSock_->send(allocTokensCmd.data(), allocTokensCmd.size(), ZMQ_SNDMORE);
       requestSock_->send((long *)(&num_tokens), sizeof(num_tokens));
       
       COUT << "zmq_scheduler::allocate_tokens SENT: num_tokens: "
          << num_tokens  << ENDL;
       
        // receive the data
       bool result = false;
       long ret = requestSock_->recv((void *)&result, sizeof(result));
       if (ret != sizeof(result)) {
          COUT << "Error: allocate_tokens does not return the result state" << ENDL;
          throw std::runtime_error("ERROR: run does not return the result state");
       }
       COUT << "zmq_scheduler::allocate_tokens: size=" << ret << ", result=" << result << ENDL;
       COUT << "zmq_scheduler:allocate_tokens END" << ENDL;
       return result;
     } catch(const std::exception &e) {
       std::cerr << "zmq_scheduler:allocate_tokens" << e.what() << std::endl;
       std::throw_with_nested( std::runtime_error("zmq_scheduler:allocate_tokens "));
       return false;
     }
   }
   //**********  END client part of the server messages tasks

  /**
  \brief Set the data element on a previously book data reference
  \tparam T Element type for the data element.
  \param item element to store at the data server.
  \param ref data_ref_type of the server and position for that data (if the ref param its not occupied it gets a new free reference on the same server), if it is null it gets a free one on the local server).
  */
  template <class T>
  data_ref_type set(T &&item, data_ref_type ref = data_ref_type{})
  {
      return data_service_->set(std::forward<T>(item), ref);
  }

  /**
  \brief Get the data element from the server and position
  referenced in the ref param.
  \tparam T Element type for the data element.
  \param ref data_ref_type of the server and position for tha data.
  */
  template <class T>
  T get (data_ref_type ref)
  {
    return data_service_->get<T>(ref,false);
  }

  /**
  \brief Get the data element from the server and position
  referenced in the ref param and release it.
  \tparam T Element type for the data element.
  \param ref data_ref_type of the server and position for tha data.
  */
  template <class T>
  T get_release (data_ref_type ref)
  {
    return data_service_->get<T>(ref,true);
  }
  
  /**
  \brief Get the data element from the server and position
  referenced in the ref param and release it after all the servers have gotten it.
  \tparam T Element type for the data element.
  \param ref data_ref_type of the server and position for tha data.
  */
  template <class T>
  T get_release_all (data_ref_type ref)
  {
    return data_service_->get<T>(ref,true,total_servers_);
  }
  
  /**
  \brief Set number of grppi threads.
  */
  void set_concurrency_degree(long degree) noexcept { concurrency_degree_ = degree; }

  /**
  \brief Get number of grppi threads.
  */
  long concurrency_degree() const noexcept { return concurrency_degree_; }


  public:
    // collection of stage functions
    std::vector<std::function<void(task_type&)>> functions;
    std::vector<long> max_num_new_tokens;

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
    const std::string finishTaskCmd{"FINISH_TSK"};
    /// tag for consume command
    const std::string allocTokensCmd{"ALLOC_TOK"};


    constexpr static long default_functions_ = 10000;
    constexpr static long default_tokens_ = 100;

    const long max_functions = default_functions_;
    const long max_tokens = default_tokens_;
    

    // construct params
    std::map<long, std::string> machines_;
    long id_{0};
    long server_id_{0};
    long schedServer_port_{0};
    bool is_sched_server_{false};
    long schedServer_portserv_port_{1};
    long max_tokens_{default_tokens_};
    long total_servers_{0};

    // array of seq. functions ids
    std::vector<long> seq_func_ids_;

    // array of new/old token functions
    std::vector<long> new_token_func_;

    // port service
    std::shared_ptr<zmq_port_service> port_service_;
    // data service
    std::shared_ptr<zmq_data_service> data_service_;

    //mutual exclusion data
    std::atomic_flag task_gen_ = ATOMIC_FLAG_INIT;
    
    // zeroMQ data
    zmq::context_t context_;
    std::unique_ptr<zmq::socket_t> replySock_;
    std::map<std::thread::id, std::shared_ptr<zmq::socket_t>> requestSockList_;
    //mutual exclusion data for socket map structure
    std::atomic_flag accessSockMap_ = ATOMIC_FLAG_INIT;

   
    /// server address
    std::thread server_thread_;
    
    // pool of threads
    long concurrency_degree_;
    mutable dist_pool<zmq_scheduler<task_type>> thread_pool_;

    // pointer to zmq_scheduler_thread
    zmq_scheduler_thread<task_type> * zmq_sched_thread_ = NULL;


    // local counter for generating task ids
    long last_local_task_id_ = 0;
    // distance of local machine on machines_ maps
    long machine_map_order_ = 0;
    // number of machines in machines_ maps
    long total_machines_ = 0;

    //mutual exclusion data
    std::atomic_flag task_ids_gen_ = ATOMIC_FLAG_INIT;

  /**
  \brief Function to create a zmq request socket for the port service
  \return Shared pointer with the zmq socket.
  */
  std::shared_ptr<zmq::socket_t> create_socket ()
  {
    COUT << "zmq_scheduler::create_socket begin" << ENDL;
   
    // create request socket shared pointer
    std::shared_ptr<zmq::socket_t> requestSock_ = std::make_shared<zmq::socket_t>(context_,ZMQ_REQ);

    // connect request socket
    std::ostringstream ss;
    ss << tcpConnectPattern[0] << machines_[server_id_] << tcpConnectPattern[1] << schedServer_port_;
    requestSock_->connect(ss.str());

    COUT << "zmq_scheduler::create_socket connect: " << ss.str() << ENDL;

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
    \param maxTokens max number of tokens to handle
    \param total_servers total number of servers to handle
    */
    zmq_scheduler_thread(long maxTokens, long total_servers):
        seq_tasks_(),
        new_tok_par_tasks_(maxTokens),
        old_tok_par_tasks_(maxTokens),
        requests_(maxTokens),
        run_requests_(total_servers),
        end_requests_(total_servers),
        blocked_servers_(total_servers) {
            COUT << "zmq_scheduler_thread::constructor maxTokens = " << maxTokens << ", total_servers = " << total_servers << ENDL;
            maxTokens_ = maxTokens;
            total_servers_ = total_servers;
            //new_tok_par_tasks_.registry(100); ///?????
        };

    /**
    \brief functor member
    \param sched pointer to the scheduler object
    */
    void operator() (zmq_scheduler<task_type> * sched)
    {
        COUT << "zmq_scheduler_thread() maxTokens = " << maxTokens_ << ", total_servers_ = " << total_servers_ << ENDL;
        server_func (sched);
    }

  private:
  
    /// server data
    long tokens_{0};
    double ratio_create_tokens_{0.5};
    long pool_threads_{0};
    long total_running_servers_{0};
    long total_ending_servers_{0};
    long total_blocked_servers_{0};
    // array of seq. functions ids
    std::vector<long> seq_func_ids_;
    // server id that executes the scheduler
    long sched_server_id_;
    // server ids for each server (server_id)
    std::vector<long> servers_ids_;
    // num. of threads for each server (server_id)
    std::vector<long> servers_num_threads_;
    // server for each sequential tasks (task.get_id(),server_id)
    std::map<long,long> seq_servers_;
    // sequential tasks queues (server_id, (task.get_id(),task))
    std::map<long,multi_queue<long,task_type>> seq_tasks_;
    // new token parallel tasks queue (list_server_id, task)
    multi_queue_hard<long,task_type> new_tok_par_tasks_;
    // old token parallel tasks queue (list_server_id, task)
    multi_queue_hard<long,task_type> old_tok_par_tasks_;
    // waiting get requests queue (server_id, client_id)
    multi_queue<long,std::string> requests_;
    // queue for pending run request (client_id)
    locked_mpmc_queue<std::string> run_requests_;
    // queue for pending end request (client_id)
    locked_mpmc_queue<std::string> end_requests_;
    // queue for blocked servers (client_id,server_id,task)
    locked_mpmc_queue<std::tuple<std::string,task_type>> blocked_servers_;
    // set of enabled sequential tasks (task.get_id())
    std::set<long> set_enabled_;
    // set of task that potentially can create new tokens (task.get_id())
    std::set<long> set_new_tok_tasks_;
    // set of task that cannot create new tokens (task.get_id())
    std::set<long> set_old_tok_tasks_;

    // list of blocked dependent task
    // map(task.get_task_id(), tuple(task,set(before_dep_task_id)))
    std::map<long,std::pair<task_type,std::set<long>>> dep_tasks_;
    
    long maxTokens_{0};
    long total_servers_{0};
    task_type last_exec_task_{};
    
    unsigned long total_func_stages_{0};

    
    /**
    \brief Multiqueues initialization.
    
    Init thoses queues and structures that do not change
    between parallel executions of the same scheduler (Carea)
    
    \param machines map of <servers ids, machine IPs>
    */
    void queues_init (std::map<long, std::string> machines)
    {
      try {
        for (const auto& elem : machines) {
            COUT << "zmq_scheduler_thread::queues_init registry = " << elem.first << ", " << elem.second << ENDL;
            new_tok_par_tasks_.registry(elem.first);
            COUT << "zmq_scheduler_thread::queues_init new_tok_par_tasks_" << ENDL;

            old_tok_par_tasks_.registry(elem.first);
            COUT << "zmq_scheduler_thread::queues_init old_tok_par_tasks_" << ENDL;

            requests_.registry(elem.first);
            COUT << "zmq_scheduler_thread::queues_init requests_" << ENDL;
        }
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::queues_init: " << e.what() << std::endl;
      }
    }
    
    /**
    \brief Server initialization.
    \param sched pointer to the scheduler object
    */
    void server_init (zmq_scheduler<task_type> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::server_init begin" << ENDL;

        // inproc server socket binded
        std::ostringstream ss;
        ss << sched->tcpBindPattern[0] << "0";
        sched->replySock_= std::make_unique<zmq::socket_t>(sched->context_,ZMQ_ROUTER);
        sched->replySock_->bind(ss.str());
        COUT << "zmq_scheduler_thread::server_init bind: " << ss.str() << ENDL;

        size_t size = 256;
        char buf[256];
        COUT << "zmq_scheduler: getsockopt" << ENDL;
        sched->replySock_->getsockopt(ZMQ_LAST_ENDPOINT,buf,&size);
        std::string address(buf);
        std::string delimiter = ":";
        long pos = address.find(delimiter, address.find(delimiter)+1)+1;
        std::string srtPort = address.substr(pos); // token is "scott"
        COUT << "zmq_scheduler_thread: " << srtPort << ENDL;
        
        long port = atoi(srtPort.c_str());
        COUT << "zmq_scheduler_thread::server_init " << address << " (" << sched->id_ << "," << port << ")" << ENDL;
        COUT << "zmq_scheduler_thread::server_init sched->port_service_->set begin" << ENDL;
        sched->port_service_->set(0,sched->schedServer_portserv_port_,port);
        COUT << "zmq_scheduler_thread::server_init sched->port_service_->set end" << ENDL;
      
        // init multiqueues
        queues_init (sched->machines_);
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::server_init: " << e.what() << std::endl;
      }
    }
    
    /**
    \brief Push a sequential task on its corresponding multique after checking and allocating.
    \param sched pointer to the scheduler object
    \param task Task to be inserted
    */
    void push_seq (zmq_scheduler<task_type> * sched, task_type && task)
    {
    try {
      COUT << "zmq_scheduler_thread::push_seq: BEGIN" << ENDL;
      // allocate and register multi_queue if not done before
      if ( (seq_tasks_.find(seq_servers_[task.get_id()]) != seq_tasks_.end()) &&
           (seq_tasks_.find(seq_servers_[task.get_id()])->
                            second.is_registered(task.get_id())) )  {
        COUT << "zmq_scheduler_thread::push_seq: NOT first for task: "<< task.get_id() << ENDL;
        auto local_ids = task.get_local_ids();
        if ( (task.get_is_hard()) &&
             (std::find(local_ids.begin(),
                        local_ids.end(),
                        seq_servers_[task.get_id()]) == local_ids.end()) ) {
          // ERROR: used seq. server is not valid for actual iteration of the task
          std::cerr << "zmq_scheduler_thread::push_seq ERROR sequential task: " << task.get_id() << " cannot execute on server " <<  seq_servers_[task.get_id()]  << std::endl;
          assert (false);
        }
      } else {
        auto local_ids = task.get_local_ids();

        COUT << "zmq_scheduler_thread::push_seq: first for task: "<< task.get_id() << ", is_hard = " << task.get_is_hard() << ", task.get_local_ids().size() = " << local_ids.size() << ", task.get_local_ids()[0] = " << local_ids[0] << ", seq_servers_[" << task.get_id() << "] = " << seq_servers_[task.get_id()] << ENDL;
        
        if ( (task.get_is_hard()) &&
             (std::find(local_ids.begin(),
                         local_ids.end(),
                         seq_servers_[task.get_id()]) == local_ids.end()) ) {
          // prev. assigned server is not adecuate for this task, change it
          COUT << "zmq_scheduler_thread::push_seq: change server " << ENDL;
          seq_servers_[task.get_id()] = local_ids[0];
        }
          
        // if multi_queue for this server do not exists, create it
        if (seq_tasks_.find(seq_servers_[task.get_id()]) == seq_tasks_.end()) {
          seq_tasks_.emplace(std::piecewise_construct,
                             std::forward_as_tuple(seq_servers_[task.get_id()]),
                             std::forward_as_tuple(sched->max_tokens_));
          COUT << "zmq_scheduler_thread::push_seq: create server queue " << seq_servers_[task.get_id()] << ENDL;
        }
        // if task is not registered in its multi_queue, register it
        if (!seq_tasks_.find(seq_servers_[task.get_id()])->
                             second.is_registered(task.get_id())) {
          seq_tasks_.find(seq_servers_[task.get_id()])->second.registry(task.get_id());
              COUT << "zmq_scheduler_thread::push_seq: register task " << task.get_id() << " on server " << seq_servers_[task.get_id()] <<  ENDL;
        }
      }
      // push task onto the multi_queue of the corresponding server
      COUT << "zmq_scheduler_thread::push_seq: push: " << task.get_id() << ENDL;
      seq_tasks_.find(seq_servers_[task.get_id()])->second.push(task.get_id(), std::move(task));
      COUT << "zmq_scheduler_thread::push_seq: END" << ENDL;
    } catch(const std::exception &e) {
      std::cerr << "zmq_scheduler_thread::push_seq: ERROR: " << e.what() << std::endl;
    }
    }
    
    /**
    \brief insert a new task on the corresponding task queues
    \param sched pointer to the scheduler object
    \param task Task to be inserted
    */
    void insert_task (zmq_scheduler<task_type> * sched, task_type && task)
    {
      // if new task is sequential, insert it on the corresponding server's queue
      if (seq_servers_.find(task.get_id()) != seq_servers_.end()) {
        push_seq (sched, std::move(task));
        COUT << "zmq_scheduler_thread::insert_task: NEW TASK SEQ" << ENDL;

      // if new task could create new tokens, insert on new_token_parallel_queue
      } else if (set_new_tok_tasks_.find(task.get_id())!=set_new_tok_tasks_.end()) {
        new_tok_par_tasks_.push(task.get_local_ids(), std::move(task), task.get_is_hard());
        COUT << "zmq_scheduler_thread::insert_task: NEW TASK NEW_TOKEN" << ENDL;
      // else insert on old_token_parallel_queue
      } else {
        old_tok_par_tasks_.push(task.get_local_ids(), std::move(task), task.get_is_hard());
        COUT << "zmq_scheduler_thread::insert_task: NEW TASK OLD_TOKEN" << ENDL;
      }
    }
  
    /**
    \brief Server function to store and release data form the storage array.
    \param sched pointer to the scheduler object
    */
    void server_func (zmq_scheduler<task_type> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::server_func begin" << ENDL;

        // initialize server
        server_init (sched);
        COUT << "zmq_scheduler_thread::server_func:  server_init end" << ENDL;

        while (1) {

          zmq::message_t msg;
          std::string client_id;

          // receive client id
          COUT << "zmq_scheduler_thread::server_func sched->replySock_->recv begin" << ENDL;
          sched->replySock_->recv(&msg);
          client_id = std::string((char *)msg.data(), msg.size());
          COUT << "zmq_scheduler_thread::server_func sched->replySock_->recv client_id" << ENDL;

          // recv zero frame
          sched->replySock_->recv(&msg);
        
          if (msg.size() != 0) {
            COUT << "zmq_scheduler_thread::server_func ERROR frame zero: " << (char *)msg.data() << "| size: " << msg.size() << ENDL;
            throw std::runtime_error ("Error frame zero");
          }

          // recv command
          sched->replySock_->recv(&msg);
          COUT << "zmq_scheduler_thread:: CMD: " << (char *)msg.data() << "| size: " << msg.size() << ENDL;

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
          } else if ( (msg.size() == sched->finishTaskCmd.size()) &&
                      (0 == std::memcmp(msg.data(),
                                    static_cast<const void*>(sched->finishTaskCmd.data()),
                                    sched->finishTaskCmd.size())) ) {
            // serving consume command
            srv_finish_task_cmd (sched, client_id);
          
          // allocate tokens command
          } else if ( (msg.size() == sched->allocTokensCmd.size()) &&
                      (0 == std::memcmp(msg.data(),
                                      static_cast<const void*>(sched->allocTokensCmd.data()),
                                      sched->allocTokensCmd.size())) ) {
            // serving consume command
            srv_allocate_tokens_cmd (sched, client_id);
          
          // ERROR
          } else {
            COUT << "zmq_scheduler_thread::server_func ERROR: Cmd not found" << ENDL;
          }
       
          //
          // server pending requests
          //
        
          // if there are pending requests -> try to assign them to pending tasks
          COUT << "zmq_scheduler_thread::server_func requests_.empty(): " << requests_.empty() << ENDL;

          if (! requests_.empty()) {

            // if no running tokens -> Check if we need to end the run,
            //                         the whole server or do nothing.
            COUT << "zmq_scheduler_thread::server_func tokens_: " << tokens_ << ENDL;

            if (tokens_ <= 0) {
              // check that all threads are stopped before finish run calls
              if (requests_.count() == pool_threads_) {
                // check run and end received messages
                bool status = finish_execution (sched);
                //if everything is done, finish the thread
                if (status) break;
              }
            } else {
              // get which servers has pending requests
              auto set_req_servers = requests_.loaded_set();
              COUT << "zmq_scheduler_thread::server_func: set_req_servers.size(): " << set_req_servers.size() << ENDL;

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
        COUT << "zmq_scheduler_thread::server_func end" << ENDL;
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::server_func: general exec: " << e.what() << std::endl;
     }

    }
 
 
    /**
    \brief End command service
    \param sched poiner to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_end_cmd (zmq_scheduler<task_type> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::end_cmd_srv BEGIN" << ENDL;

        // receive server id
        zmq::message_t msg;
        sched->replySock_->recv(&msg);
        //long server_id = *((long*) msg.data());

        // store end request to wake it up at the end, increase ending servers
        end_requests_.push(client_id);
        total_ending_servers_++;
        COUT << "zmq_scheduler_thread::srv_end_cmd: END total_ending_servers_=" << total_ending_servers_ << ENDL;
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_end_cmd" << e.what() << std::endl;
      }
    }
    
    /**
    \brief run command service
    \param sched poiner to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_run_cmd (zmq_scheduler<task_type> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_run_cmd BEGIN" << ENDL;

        // recv number of threads
        zmq::message_t msg;
        sched->replySock_->recv(&msg);
        long num_threads = *((long*) msg.data());
        servers_num_threads_.push_back(num_threads);
    
        COUT << "zmq_scheduler_thread::srv_run_cmd num_threads = " << num_threads << ENDL;

        // receive server id
        sched->replySock_->recv(&msg);
        long server_id = *((long*) msg.data());
        servers_ids_.push_back(server_id);

        COUT << "zmq_scheduler_thread::srv_run_cmd server_id = " << server_id << ENDL;

        // receive if the server is the scheduler server
        sched->replySock_->recv(&msg);
        bool is_sched_server = ((*((long*) msg.data())) > 0);

        COUT << "zmq_scheduler_thread::srv_run_cmd is_sched_server = " << is_sched_server << ENDL;

        if (is_sched_server) {
          // set it as the sched server
          sched_server_id_=server_id;
          // get list of seq. functions ids
          sched->replySock_->recv(&msg);
          seq_func_ids_.resize(msg.size()/sizeof(long));
          std::memcpy(seq_func_ids_.data(), msg.data(), msg.size());
           
          COUT << "zmq_scheduler_thread::srv_run_cmd seq_func_ids_.size() = " << seq_func_ids_.size() << ENDL;

          // get list of new token functions
          sched->replySock_->recv(&msg);
          total_func_stages_ = msg.size()/sizeof(long);
          std::vector<long> new_token_func_(total_func_stages_);
          std::memcpy(new_token_func_.data(), msg.data(), msg.size());
            
          COUT << "zmq_scheduler_thread::srv_run_cmd new_token_func_.size() = " << new_token_func_.size() << ENDL;

            // set new and ols tokens masks
          for (unsigned long i=0; i<new_token_func_.size(); i++) {
            COUT << "zmq_scheduler_thread::srv_run_cmd new_token_func_[" << i << "] = " << new_token_func_[i] << ENDL;

            if (new_token_func_[i] == 1) {
              set_new_tok_tasks_.insert(i);
            } else {
              set_old_tok_tasks_.insert(i);
            }
          }
          COUT << "zmq_scheduler_thread::srv_run_cmd:  set_new_tok_tasks_.size() = " << set_new_tok_tasks_.size() << ", set_old_tok_tasks_.size() = " << set_old_tok_tasks_.size() << ENDL;
          for (const auto &aux : set_new_tok_tasks_)  {
            COUT << "zmq_scheduler_thread::srv_run_cmd set_new_tok_tasks_  elem = " << aux << ENDL;
          }
          for (const auto &aux : set_old_tok_tasks_)  {
            COUT << "zmq_scheduler_thread::srv_run_cmd set_old_tok_tasks_  elem = " << aux << ENDL;
          }
        }

        // store run request to wake it up at the end, increase running servers and threads
        run_requests_.push(client_id);
        total_running_servers_++;
        pool_threads_= pool_threads_ + num_threads;

        COUT << "zmq_scheduler_thread::srv_run_cmd total_running_servers_ = " << total_running_servers_ << ENDL;
        COUT << "zmq_scheduler_thread::srv_run_cmd sched->total_servers_ = " << sched->total_servers_ << ENDL;

        // if all run requests have arrived, launch initial task
        // NOTE: inital task is supposed to create new tokens???
        if (total_running_servers_ == sched->total_servers_) {
        
          // fill up asignation of sequential tasks to servers.
          // also enable sequential tasks.
          unsigned long indx = 0;
          for (unsigned long i=0; i<seq_func_ids_.size(); i++) {
            seq_servers_[seq_func_ids_[i]]=servers_ids_[indx];
            set_enabled_.insert(seq_func_ids_[i]);
            indx++;
            if (indx >=servers_ids_.size()) indx=0;
          }
          COUT << "zmq_scheduler_thread::srv_run_cmd seq_servers_.size() = " << seq_servers_.size() << ENDL;

          COUT << "zmq_scheduler_thread::srv_run_cmd set_enabled_.size() = " << set_enabled_.size() << ENDL;
          for (const auto &aux : set_enabled_)  {
            COUT << "zmq_scheduler_thread::srv_run_cmd set_enabled_  elem = " << aux << ", initial server = " << seq_servers_[aux] << ENDL;
          }

          COUT << "zmq_scheduler_thread::srv_run_cmd servers_ids_.size() = " << servers_ids_.size() << ENDL;
            
          task_type task_seq = task_type{0,0,{0,0,0},std::vector<long>{servers_ids_[0]},false};
          COUT << "zmq_scheduler_thread::srv_run_cmd created init task = {0,0,0,{" << servers_ids_[0] << "},false}" << ENDL;
          if (seq_servers_.find(0) != seq_servers_.end()) {

            //run initial seq task (0) on its corresp. server
            push_seq (sched, std::move(task_seq));
            COUT << "zmq_scheduler_thread::srv_run_cmd INIT SEQ = " << ENDL;
          } else {
            // run initial task on first server
            new_tok_par_tasks_.push(task_seq.get_local_ids(), std::move(task_seq), task_seq.get_is_hard());
            COUT << "zmq_scheduler_thread::srv_run_cmd INIT PAR = " << ENDL;
          }

	  //--------------------------------------------
	  // This is not true for the map-reduce
          //assert(tokens_==0);
          tokens_+=1;
	  //-------------------------------------------
          COUT << "zmq_scheduler_thread::end_cmd_srv END" << ENDL;
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
    void srv_get_cmd (zmq_scheduler<task_type> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_get_cmd  BEGIN" << ENDL;
        
        // recv server_id and task_id
        zmq::message_t msg;

        sched->replySock_->recv(&msg);
        long server_id = *((long*) msg.data());

        task_type old_task{};
        old_task.recv(*(sched->replySock_));
          
        COUT << "zmq_scheduler_thread::srv_get_cmd: server_id: "
                  << server_id << ", old_task: (" << old_task.get_id()
                  << ", " << old_task.get_task_id() << ")" << ENDL;

        // if a sequential task ended, enable it
        if ( (old_task.get_id() >= 0) &&
             (seq_servers_.find(old_task.get_id()) != seq_servers_.end() ) ) {
          set_enabled_.insert(old_task.get_id());
          COUT << "zmq_scheduler_thread::srv_get_cmd: seq task ended" << ENDL;
        }

        // if the executed task is not null, save it as last executed task
        if (old_task != task_type{}) {
            last_exec_task_ = std::move(old_task);
        }

        // push task request to be served latter
        COUT << "zmq_scheduler_thread::srv_get_cmd: push request" << ENDL;
        requests_.push(server_id,client_id);
        COUT << "zmq_scheduler_thread::srv_get_cmd: END" << ENDL;
          
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
    bool srv_set_cmd (zmq_scheduler<task_type> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_set_cmd  BEGIN" << ENDL;
          
        // recv server_id, task and new_token
        zmq::message_t msg;

        sched->replySock_->recv(&msg);
        COUT << "zmq_scheduler_thread::srv_set_cmd  recv_ B" << ENDL;
        long server_id = *((long*) msg.data());

        task_type task{};
        task.recv(*(sched->replySock_));
        COUT << "zmq_scheduler_thread::srv_set_cmd  recv_ C" << ENDL;

        sched->replySock_->recv(&msg);
        COUT << "zmq_scheduler_thread::srv_set_cmd  recv_ D" << ENDL;
        bool new_token = ( *((long*) msg.data()) > 0 );

        COUT << "zmq_scheduler_thread::srv_set_cmd: server_id: "
                  << server_id << ", task: (" << task.get_id()
                  << ", " << task.get_task_id() << ")" << ENDL;
          
        
        // if task requires a new token check if possible and then increase them
        if (new_token) {
          if (tokens_ >= sched->max_tokens_) {
            // if tokens can't be increased block the server until they can
            blocked_servers_.push(make_tuple(client_id,std::move(task)));
            total_blocked_servers_++;

            COUT << "zmq_scheduler_thread::srv_set_cmd: total_blocked_servers_" << total_blocked_servers_ << ENDL;

            if (total_blocked_servers_ >= sched->total_servers_) {
              // if all servers are blocked, launch a run-time exception.
              throw std::runtime_error ("All servers blocked");
            }
            // stop and jump to read next request
            COUT << "zmq_scheduler_thread::srv_set_cmd: continue"  << ENDL;
            return (true);
            
          }
          tokens_++;
          COUT << "zmq_scheduler_thread::srv_set_cmd: tokens_" << tokens_ << ENDL;

        }
        
        
        // if task requires dependencies store it on dependable task list
        auto before_dep = task.get_before_dep();
        if (! before_dep.empty()) {
          // insert task on depedent task list (it shouldn't exist)
          assert(dep_tasks_.find(task.get_task_id()) == dep_tasks_.end());
          dep_tasks_.emplace(std::piecewise_construct,
                             std::forward_as_tuple(task.get_task_id()),
                             std::forward_as_tuple(                              make_pair(std::move(task),before_dep)));
          COUT << "zmq_scheduler_thread::srv_set_cmd Inserted dependant task: task.get_task_id() = " << task.get_task_id() << ENDL;
        } else {
          // insert task in its corresponding queue.
          COUT << "zmq_scheduler_thread::srv_set_cmd: check and insert task" << ENDL;
          insert_task (sched, std::move(task));
        }
        // send back an ACK
        sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
        sched->replySock_->send("", 0, ZMQ_SNDMORE);
        sched->replySock_->send("", 0);
       
        COUT << "zmq_scheduler_thread::srv_set_cmd END" << ENDL;
        
       } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_set_cmd ERROR: " << e.what() << std::endl;
      }
      return (false);
    }

    /**
    \brief finish task command service
    \param sched pointer to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_finish_task_cmd (zmq_scheduler<task_type> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_finish_task_cmd  BEGIN" << ENDL;
        
        // recv server_id and task_id
        zmq::message_t msg;
        
        task_type old_task{};
        old_task.recv(*(sched->replySock_));

        sched->replySock_->recv(&msg);
        long num_free_tokens = *((long*) msg.data());

        COUT << "zmq_scheduler_thread::srv_finish_task_cmd: task = " << old_task.get_task_id() << " has consume " <<  num_free_tokens << " tokens" << ENDL;

        //check if old task has after dependancies
        auto after_dep = old_task.get_after_dep();
        if (! after_dep.empty()) {
            //eliminate finished dependencies
            COUT << "zmq_scheduler_thread::srv_finish_task_cmd: remove dependencies" << ENDL;
            for (auto it_dep=after_dep.begin(); it_dep!=after_dep.end(); it_dep++) {
              // remove dependency from one task
              auto it_task = dep_tasks_.find(*it_dep);
              // check that the dependent task exist)
              assert(it_task != dep_tasks_.end());
              // check there is one an only one dep to erase
              long result = it_task->second.second.erase(old_task.get_task_id());
              assert(result == 1);
              COUT << "zmq_scheduler_thread::srv_finish_task_cmd: task = " << it_task->second.first.get_task_id() << " remove task_dep = " << old_task.get_task_id() << " it_task->second.second.size() = " << it_task->second.second.size() << " it_task->second.second.(0) = " << ((!it_task->second.second.empty())?*(it_task->second.second.begin()):-1) << ENDL;

              if (it_task->second.second.empty()) {
                //  no more dependencies, remove and insert on tasks queues
                COUT << "zmq_scheduler_thread::srv_finish_task_cmd: task = " << it_task->second.first.get_task_id() << " ready to execute" << ENDL;

                // insert block task into the corresponding queues
                insert_task (sched, std::move(it_task->second.first));
                
                // erase task from dependency map
                dep_tasks_.erase(it_task);
              }
            }
        }
        
        // free tokens
        assert (tokens_ >= num_free_tokens);
        tokens_ = tokens_ - num_free_tokens;
        COUT << "zmq_scheduler_thread::srv_finish_task_cmd: tokens_ before unblock tasks = " << tokens_ << ENDL;

        // try to unblock as many blocked servers as possible
        while ( (! blocked_servers_.empty()) &&
                (tokens_ < sched->max_tokens_) ) {
          // insert new task from blocked server and wake it up
          auto data = blocked_servers_.pop();
          auto blk_client_id = std::get<0>(data);
          auto blk_task = std::get<1>(data);
          total_blocked_servers_--;

          COUT << "zmq_scheduler_thread::srv_finish_task_cmd: total_blocked_servers_" << total_blocked_servers_ << ENDL;

          // insert block task into the corresponding queues
          insert_task (sched, std::move(blk_task));

          // send back an ACK to the blocked server
          sched->replySock_->send(blk_client_id.data(),
                                 blk_client_id.size(),
                                 ZMQ_SNDMORE);
          sched->replySock_->send("", 0, ZMQ_SNDMORE);
          sched->replySock_->send("", 0);
          COUT << "zmq_scheduler_thread::srv_finish_task_cmd: ACK SENT for SET TASK" << ENDL;
          
          // update tokens
          tokens_++;
        }
        
        COUT << "zmq_scheduler_thread::srv_finish_task_cmd: tokens_ at the end = " << tokens_ << ENDL;

        // send back an ACK for consume request
        sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
        sched->replySock_->send("", 0, ZMQ_SNDMORE);
        sched->replySock_->send("", 0);
        COUT << "zmq_scheduler_thread::srv_finish_task_cmd: ACK SENT " << ENDL;

       } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_finish_task_cmd: " << e.what() << std::endl;
      }
    }

    /**
    \brief pre-allocate a number of tokens if possible
    \param sched pointer to the scheduler object
    \param client_id zmq client identificator for answering command.
    */
    void srv_allocate_tokens_cmd (zmq_scheduler<task_type> * sched, std::string client_id)
    {
      try {
        COUT << "zmq_scheduler_thread::srv_allocate_tokens_cmd  BEGIN" << ENDL;
        
        // recv server_id and task_id
        zmq::message_t msg;
        sched->replySock_->recv(&msg);
        long num_tokens = *((long*) msg.data());
        COUT << "zmq_scheduler_thread::srv_allocate_tokens_cmd  tokens to allocate = "<< num_tokens << ENDL;

        // Allocate the requested tokens if possible
        bool result = false;
        if ( (tokens_ + num_tokens) <= sched->max_tokens_ ) {
          tokens_ = tokens_ + num_tokens;
          result = true;
        }

        COUT << "zmq_scheduler_thread::srv_allocate_tokens_cmd  result = "<< result << "total_tokens = " << tokens_<< ENDL;

        // send back an ACK for consume request
        sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
        sched->replySock_->send("", 0, ZMQ_SNDMORE);
        sched->replySock_->send((bool *)(&result), sizeof(result));
        COUT << "zmq_scheduler_thread::srv_allocate_tokens_cmd: Result SENT " << ENDL;

       } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::srv_allocate_tokens_cmd: " << e.what() << std::endl;
      }
    }

    /**
    \brief Finalize execution (treats run and end messages received)
    \param sched pointer to the scheduler object
    \return booolean value true -> Finish server thread; false -> continue normal execution.
    */
    bool finish_execution (zmq_scheduler<task_type> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::finish_execution  BEGIN" << ENDL;

        // if run request are all received then finish them
        // note: run request can not be all recieved with no tokens
        //       unless is the end of a run
        //       (when last run request arrive, tokens is set to 1 immediately)
        COUT << "zmq_scheduler_thread::finish_execution total_running_servers_: " << total_running_servers_ << ", sched->total_servers_: " << sched->total_servers_ << ENDL;
        if (total_running_servers_ == sched->total_servers_) {
          while (! run_requests_.empty()) {
            // get run reply address and send back and ACK
            auto client_id = run_requests_.pop();
            sched->replySock_->send(client_id.data(),
                                    client_id.size(), ZMQ_SNDMORE);
            sched->replySock_->send("", 0, ZMQ_SNDMORE);
            last_exec_task_.send(*(sched->replySock_));
            COUT << "zmq_scheduler_thread::finish_execution SENT RUN ACK "<< ENDL;
          }

          // reset counter of running servers
          total_running_servers_ = 0;
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
        COUT << "zmq_scheduler_thread::finish_execution total_running_servers_: " << total_running_servers_ << ", total_ending_servers_: " << total_ending_servers_ << ", sched->total_servers_: " << sched->total_servers_ << ", pool_threads_: " << pool_threads_ <<  ENDL;
        if ( (total_running_servers_ == 0)  &&
             (total_ending_servers_ == sched->total_servers_) &&
             (requests_.count() == pool_threads_) ) {
          // finish get task request
          // send a null task to all pending requests
          while (! requests_.empty()) {
            auto client_id = requests_.pop();
            sched->replySock_->send(client_id.data(),
                                    client_id.size(), ZMQ_SNDMORE);
            sched->replySock_->send("", 0, ZMQ_SNDMORE);
            task_type task{}; // default/empty task
            task.send(*(sched->replySock_));
              
            // decrease running threads.
            // If all done wake up all end replies and end sched. thread
            pool_threads_--;
            COUT << "zmq_scheduler_thread::finish_execution: SENT GET TASK ACK NULL: task: (" << task.get_id() << ", " << task.get_task_id() << ")" << ENDL;

          }
          COUT << "zmq_scheduler_thread::finish_execution: pool_threads_ = "<< pool_threads_ << ENDL;

          if (pool_threads_ <= 0) {
            while (! end_requests_.empty()) {
              // get run reply address and send back and ACK
              auto client_id = end_requests_.pop();
              sched->replySock_->send(client_id.data(),
                                      client_id.size(), ZMQ_SNDMORE);
              sched->replySock_->send("", 0, ZMQ_SNDMORE);
              sched->replySock_->send("", 0);
              COUT << "zmq_scheduler_thread::finish_execution SENT END ACK " << ENDL;

            }
            // reset counter of ending servers
            total_ending_servers_ = 0;
                  
            //
            // end Scheduling thread
            //
            return (true);
          }
        }
      COUT << "zmq_scheduler_thread::finish_execution  END" << ENDL;
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::finish_execution: " << e.what() << std::endl;
      }
      return (false);
    }

    /**
    \brief Check sequential tasks for all servers with pending requests.
    \param sched pointer to the scheduler object
    \param set_req_servers set of servers with pending get requests
    */
    void exec_seq_task (zmq_scheduler<task_type> * sched,
                        std::set<long> &set_req_servers)
    {
      try {
        static long index_new = 0;
        static long index_old = 0;
      
        COUT << "zmq_scheduler_thread::exec_seq_task BEGIN" << ENDL;

        long index = 0;
        auto it=set_req_servers.begin();
        while (it!=set_req_servers.end()) {

          //if req server do not have seq. task subqueue, go to next server
          if (seq_tasks_.find(*it) == seq_tasks_.end()) {
            COUT << "zmq_scheduler_thread::exec_seq_task seq server do not have any seq task asigned: " << *it << " server" << ENDL;
            it++;
            continue;
          }

          COUT << "zmq_scheduler_thread::exec_seq_task try " << *it << " server" << ENDL;

          long search_end = false;
          do {

            auto set_seq_task_new =
               seq_tasks_.find(*it)->second.loaded_set(set_enabled_,set_new_tok_tasks_);
            auto set_seq_task_old =
               seq_tasks_.find(*it)->second.loaded_set(set_enabled_,set_old_tok_tasks_);

            COUT << "zmq_scheduler_thread::exec_seq_task: check new seq server with request: set_seq_task_new.size() = " << set_seq_task_new.size() << ", set_seq_task_old.size() = " << set_seq_task_old.size() << ENDL;

            double actual_ratio = ((double)tokens_) / ((double) sched->max_tokens_);
            COUT << "zmq_scheduler_thread::exec_seq_task: actual_ratio = " << actual_ratio << ", ratio_create_tokens_ = " << ratio_create_tokens_ << ENDL;

            // if tokens ratio is below the threshold and there are new token tasks, select one
            if ( (! set_seq_task_new.empty()) &&
                 ( (set_seq_task_old.empty()) ||
                   (actual_ratio < ratio_create_tokens_) ) ) {
            
              COUT << "zmq_scheduler_thread::exec_seq_task: get seq new token" << ENDL;

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
              
              COUT << "zmq_scheduler_thread::exec_seq_task: get seq old token" << ENDL;

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
              COUT << "zmq_scheduler_thread::exec_seq_task: A- search_end = TRUE" << ENDL;
              it++;
              search_end = true;
            }
            
            // if an index have been found
            if (! search_end) {
              // get the task, the request and reply it
              task_type task = seq_tasks_.find(*it)->second.pop(index);
              auto client_id = requests_.pop(*it);
            
              COUT   << "zmq_scheduler_thread::exec_seq_task: id_: "
                        << *it << ", task: (" << task.get_id()
                        << ", " << task.get_task_id() << ")" << ENDL;

              sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
              sched->replySock_->send("", 0, ZMQ_SNDMORE);
              task.send(*(sched->replySock_));

              COUT << "zmq_scheduler_thread::exec_seq_task: SENT GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")" << ENDL;
              
              // if no more requests for this server, erase server from set and end the search
              if (requests_.empty(*it)) {
                it = set_req_servers.erase(it);
                search_end = true;
                COUT << "zmq_scheduler_thread::exec_seq_task: B- search_end = TRUE" << ENDL;

              }
            }
          } while (! search_end);
        }
      COUT << "zmq_scheduler_thread::exec_seq_task END" << ENDL;
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::exec_seq_task: " << e.what() << std::endl;
      }
    }
  
    /**
    \brief Check parallel tasks with data on the same server for all remaining requests
    \param sched pointer to the scheduler object
    \param set_req_servers set of servers with pending get requests
    */
    void exec_par_task_same_server (zmq_scheduler<task_type> * sched,
                                    std::set<long> &set_req_servers)
    {
      try {
        COUT << "zmq_scheduler_thread::exec_par_task_same_server BEGIN" << ENDL;

        auto it=set_req_servers.begin();
        while (it!=set_req_servers.end()) {
          COUT << "zmq_scheduler_thread::exec_par_task_same_server: check new server with request" << ENDL;

          long search_end = false;
          do {
            task_type task{};
            double actual_ratio = ((double)tokens_) / ((double) sched->max_tokens_);
          
            COUT << "zmq_scheduler_thread::exec_par_task_same_server: actual_ratio = " << actual_ratio << ", ratio_create_tokens_ = " << ratio_create_tokens_ << ENDL;

            // if there are new token parallel tasks, select one
            if ( (! new_tok_par_tasks_.empty(*it)) &&
                 ( (old_tok_par_tasks_.empty(*it)) ||
                   (actual_ratio < ratio_create_tokens_) ) ) {
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: server new token req" << ENDL;

              task = new_tok_par_tasks_.pop(*it);
            
            // else if there are old token parallel tasks, select one
            } else if (! old_tok_par_tasks_.empty(*it)) {
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: server old token req" << ENDL;
              task = old_tok_par_tasks_.pop(*it);
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: server old token req END" << ENDL;

            // else end checking parallel tasks for this server
            } else {
              it++;
              search_end = true;
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: A- search_end = TRUE" << ENDL;

            }
            
            // if a task have been found
            if (! search_end) {
              COUT << "zmq_scheduler_thread::exec_par_task_same_server: START SENDING GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")" << ENDL;

              // get the task, the request and reply it
              auto client_id = requests_.pop(*it);
              sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
              sched->replySock_->send("", 0, ZMQ_SNDMORE);
              task.send(*(sched->replySock_));

              COUT << "zmq_scheduler_thread::exec_par_task_same_server: SENT GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")" << ENDL;


              // if no more requests for this server, erase server from set and end the search
              if (requests_.empty(*it)) {
                it = set_req_servers.erase(it);
                search_end = true;
                COUT << "zmq_scheduler_thread::exec_par_task_same_server: B- search_end = TRUE" << ENDL;

              }
            }
          } while (! search_end);
        }
        COUT << "zmq_scheduler_thread::exec_par_task_same_server END" << ENDL;

      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::exec_par_task_same_server: " << e.what() << std::endl;
      }
    }

  
    /**
    \brief Check remaining parallel tasks for all remaining requests
    \param sched pointer to the scheduler object
    */
    void exec_par_task_diff_server (zmq_scheduler<task_type> * sched)
    {
      try {
        COUT << "zmq_scheduler_thread::exec_par_task_diff_server BEGIN" << ENDL;
 
        while (! requests_.empty()) {
          COUT << "zmq_scheduler_thread::exec_par_task_diff_server: check new request" << ENDL;
          task_type task{};
          double actual_ratio = ((double)tokens_) / ((double) sched->max_tokens_);
          // if there are new token parallel tasks, select one
          COUT << "zmq_scheduler_thread::exec_par_task_diff_server: actual_ratio = " << actual_ratio << ", ratio_create_tokens_ = " << ratio_create_tokens_ << ENDL;
            if ( (! new_tok_par_tasks_.empty()) &&
                 ( (old_tok_par_tasks_.empty()) ||
                   (actual_ratio < ratio_create_tokens_) ) ) {
            COUT << "zmq_scheduler_thread::exec_par_task_diff_server: server new token req" << ENDL;
            task = new_tok_par_tasks_.pop();
        
            // else if there are old token parallel tasks, select one
          } else if (! old_tok_par_tasks_.empty()) {
            COUT << "zmq_scheduler_thread::exec_par_task_diff_server: server old token req" << ENDL;
            task = old_tok_par_tasks_.pop();
            
          // else end checking parallel tasks
          } else {
            COUT << "zmq_scheduler_thread::exec_par_task_diff_server: No more request served" << ENDL;

            break;
          }
        
          // get the task, the request and reply it
          auto client_id = requests_.pop();
          sched->replySock_->send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
          sched->replySock_->send("", 0, ZMQ_SNDMORE);
          task.send(*(sched->replySock_));

          COUT << "zmq_scheduler_thread::exec_par_task_diff_server: SENT GET TASK ACK: task: (" << task.get_id() << ", " << task.get_task_id() << ")" << ENDL;
        }
        COUT << "zmq_scheduler_thread::exec_par_task_diff_server END" << ENDL;
      
      } catch(const std::exception &e) {
        std::cerr << "zmq_scheduler_thread::exec_par_task_diff_server: " << e.what() << std::endl;
      }
    }
};


template <typename task_type>
void zmq_scheduler<task_type>::launch_thread() {
    zmq_sched_thread_ = new zmq_scheduler_thread<task_type>{max_tokens_, total_servers_};
    server_thread_ = std::thread(std::ref(*zmq_sched_thread_),this);
}

}
#endif
