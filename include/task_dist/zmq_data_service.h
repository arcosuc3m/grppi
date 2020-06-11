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

#ifndef GRPPI_ZMQ_DATA_SERVICE_H
#define GRPPI_ZMQ_DATA_SERVICE_H

#include <cassert>
#include <exception>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>
#include <atomic>
#include <typeinfo>

#include <zmq.hpp>

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/any.hpp>
#include <boost/circular_buffer.hpp>
//#pragma GCC diagnostic pop

#include "zmq_port_service.h"
#include "zmq_data_reference.h"
#include "zmq_serialization.h"

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

/**
\defgroup data_service zmq_data_service
\brief Data service support types.
@{
*/
class zmq_data_service {
public:
  
  using data_ref_type = zmq_data_reference;
   
  // no copy constructors
  zmq_data_service(const zmq_data_service&) =delete;
  zmq_data_service& operator=(zmq_data_service const&) =delete;

  /**
  \brief Data Service contructor interface. Also creates one data server per node.
  \param numTokens num. of maximum elements to store
  */
  zmq_data_service(long numTokens) :
    //machines_(1,default_machine_),
    machines_{ {0,"localhost"} },
    id_(default_id_),
    dataServer_port_(port_service_->new_port()),
    numTokens_(numTokens),
    port_service_()
  {
      COUT << "zmq_data_service::zmq_data_service " << machines_[0] << ENDL;
      init_data_service ();
  }

  zmq_data_service() :
    zmq_data_service (default_numTokens_) {}
  
  /**
  \brief Data Service contructor interface. Also creates one data server per node.
  \param machines list of machine nodes.
  \param id position of this node on the machine list.
  \param port_service service to store and retrive comunication ports.
  \param numTokens num. of maximum elements to store
  */
  zmq_data_service(std::map<long, std::string> machines, long id,
              std::shared_ptr<zmq_port_service> port_service, long numTokens) :
    machines_(machines.begin(), machines.end()),
    id_(id),
    dataServer_port_(port_service->new_port()),
    numTokens_(numTokens),
    port_service_(port_service)
  {
    init_data_service ();
  }


  zmq_data_service(std::map<long, std::string> machines, long id,
              std::shared_ptr<zmq_port_service> port_service) :
    zmq_data_service (machines, id, port_service, default_numTokens_) {}

  /**
  \brief Data Server destructor. Also finish the data server if exists.
  \param machines list of machine nodes.
  \param id position of this node on the machine list.
  \param numTokens num. of maximum elements to store
  */
  ~zmq_data_service()
  {
    COUT << "zmq_data_service::~zmq_data_service begin" << ENDL;
    if (isServer_) {
        COUT << "join proxy_thread_" << ENDL;
        servers_.at(id_).first.send(endCmd.data(), endCmd.size(), 0);
        server_thread_.join();
    }
    servers_.clear();
    localServer_.clear();
    COUT << "zmq_data_service::~zmq_data_service end" << ENDL;
  }

  /**
  \brief Get the data element from the server and position
  referenced in the ref param.
  \tparam T Element type for the data element.
  \param ref data_ref_type of the server and position for tha data.
  \param release flag to release the data after getting it.
  \param release_count number of release accesses before actually releasing it.
  */
  template <class T>
  T get (data_ref_type ref, bool release=false, long release_count=1)
  {
    long send_id = ref.get_id();
    COUT << "get send id:" << send_id << ENDL;

    // lock access to this socket
    servers_.at(send_id).second.second.lock();

    // send the command tag
    servers_.at(send_id).first.send(getCmd.data(), getCmd.size(), ZMQ_SNDMORE);
    COUT << "get send cmd GET" << ENDL;

    // send the reference (server_id,pos), release flag and release count
    ref.send(servers_.at(send_id).first, ZMQ_SNDMORE);
    servers_.at(send_id).first.send((bool *)(&release), sizeof(release), ZMQ_SNDMORE);
    servers_.at(send_id).first.send((long *)(&release_count), sizeof(release_count));
    COUT << "zmq_data_service::get send ref: (" << ref.get_id() << "," << ref.get_pos() << "), release: " << release << ", release_count: " << release_count << ENDL;

    // receive the data
    zmq::message_t message;
    servers_.at(send_id).first.recv(&message);
    COUT << "zmq_data_service::get rec data: size=" << message.size() << ENDL;

    if (message.size() == 0) {
        // unlock access to this socket
        servers_.at(send_id).second.second.unlock();
        COUT << "zmq_data_service::get Error Item not found" << ENDL;
        throw std::runtime_error("zmq_data_service::get Item not found");
    }

    // receive erase flag
    bool erase_local = *((bool*) message.data());
    // receive local flag
    servers_.at(send_id).first.recv(&message);
    bool recv_local = *((bool*) message.data());
    assert (recv_local == servers_.at(send_id).second.first);
    COUT << "zmq_data_service::get erase_local=" << erase_local << ", recv_local=" << recv_local << ENDL;

    // receive data (remote or pointer to local)
    servers_.at(send_id).first.recv(&message);
    COUT << "zmq_data_service::get data received message.size()=" << message.size() << ENDL;

    T item;
    // if local data get local pointer
    if (true == recv_local) {
      // get pointer to local data slot
      shared_data_type *data = *((shared_data_type**) message.data());
      COUT << "zmq_data_service::get boost::any type=" <<  data << ENDL;//std::get<2>(*data).type().name() << ", type=" << typeid(item).name() << ENDL;

      try {
        if (std::get<2>(*data).empty()) {
          // wrap buffer inside a stream and deserialize serial_str into obj
          item = internal::template deserialize<T>((char *)std::get<1>(*data).data(), std::get<1>(*data).size());
        } else {
          auto ptr_item = boost::any_cast<std::shared_ptr<T>>(std::get<2>(*data));
          COUT << "zmq_data_service::get ptr_item=" << typeid(ptr_item).name() << ENDL;

          //move if data is to be erase
          if ( true == erase_local ) {
            // move data to item and clear any store
            item = std::move(*ptr_item);
            //data->second.reset();
          } else {
            // copy data to item
            item = *ptr_item;
          }
        }
        // release atomic_flag get on the server
        std::get<0>(*data).clear();
      } catch (...) {
        // release atomic_flag get on the server
        std::get<0>(*data).clear();
        // unlock access to this socket
        servers_.at(send_id).second.second.unlock();
        COUT << "zmq_data_service::get Exception Incorrect Type" << ENDL;
        std::throw_with_nested( std::runtime_error("zmq_data_service::get Incorrect Type"));
      }
    } else {
      try {
        // wrap buffer inside a stream and deserialize serial_str into obj
        item = internal::template deserialize<T>((char *)message.data(), message.size());
        COUT << "zmq_data_service::get deserialize item" << ENDL;

      } catch (...) {
        // unlock access to this socket
        servers_.at(send_id).second.second.unlock();
        COUT << "zmq_data_service::get Exception Incorrect Type" << ENDL;
        std::throw_with_nested( std::runtime_error("zmq_data_service::get Incorrect Type"));
      }
    }
    // unlock access to this socket
    servers_.at(send_id).second.second.unlock();
    return item;
  }

  /**
  \brief Set the data element to a server and position
  returned in the ref param.
  \tparam T Element type for the data element.
  \param item element to store at the data server.
  \param ref_in referenceof a booked server and position (default choose the first free one).
  \return final reference for the used server and position.
  */
  template <class T>
  data_ref_type set(T &&item, data_ref_type ref_in = data_ref_type{})
  {

    // if ref_in is valid  uses it server, if not uses local server
    long send_id = server_id_;
    if (ref_in != data_ref_type{}) {
        send_id = ref_in.get_id();
    }
    COUT << "zmq_data_service::set send id:" << send_id << ENDL;

    // serialize obj into an std::string only if it is not local
    std::vector<char> serial_vec;
    if (false == servers_.at(send_id).second.first) {
      try {
          serial_vec = internal::serialize(item);
      } catch (...) {
          COUT << "zmq_data_service::set Exception Type not serializable" << ENDL;
          std::throw_with_nested( std::runtime_error("zmq_data_service::set Type not serializable"));
      }
    }
    // lock access to this socket
    servers_.at(send_id).second.second.lock();

    // send the command tag
    servers_.at(send_id).first.send(setCmd.data(), setCmd.size(), ZMQ_SNDMORE);
    COUT << "zmq_data_service::set send cmd SET" << ENDL;

    // send the data reference even if it is null
    ref_in.send(servers_.at(send_id).first, ZMQ_SNDMORE);
    COUT << "zmq_data_service::set send data reference: ref: (" << ref_in.get_id() << "," << ref_in.get_pos() << ")" << ENDL;
    
    if (false == servers_.at(send_id).second.first) {
        // send local flag and data if it is NOT local data
        servers_.at(send_id).first.send((bool *)(&(servers_.at(send_id).second.first)), sizeof(bool), ZMQ_SNDMORE);
        servers_.at(send_id).first.send(serial_vec.data(), serial_vec.size());
        COUT << "zmq_data_service::set send data: size=" << serial_vec.size() << ", is_local=" << servers_.at(send_id).second.first <<ENDL;
    } else {
        // send only the local flag if it is local
        servers_.at(send_id).first.send((bool *)(&(servers_.at(send_id).second.first)), sizeof(bool));
        COUT << "zmq_data_service::set send data: is_local=" << servers_.at(send_id).second.first <<ENDL;
    }
   
    // receive the reference (server_id,pos)
    data_ref_type ref_out{};
    try {
      ref_out.recv(servers_.at(send_id).first);
    } catch(...) {
      COUT << "zmq_data_service::set ERROR: does not return a data ref" << ENDL;
      servers_.at(send_id).second.second.unlock();
      std::throw_with_nested(std::runtime_error("zmq_data_service::set does not return a data ref"));
    }
    // get local_flag
    zmq::message_t message;
    servers_.at(send_id).first.recv(&message);
    bool recv_local = *((bool*) message.data());
    assert (recv_local == servers_.at(send_id).second.first);
    COUT << "zmq_data_service::set recv ref: (" << ref_out.get_id() << "," << ref_out.get_pos() << "). local_flag = " << recv_local << ENDL;
    // if local data get local pointer
    shared_data_type *data = nullptr;
    if (true == recv_local) {
        servers_.at(send_id).first.recv(&message);
        data = *((shared_data_type**) message.data());
        COUT << "zmq_data_service::set recv shared_data_ptr=" << data << ENDL;
    }

    // check if ref is null -< error
    if (ref_out == data_ref_type{}) {
        // unlock access to this socket
        servers_.at(send_id).second.second.unlock();
        COUT << "zmq_data_service::set Error full data storage" << ENDL;
        throw std::runtime_error("zmq_data_service::set Full Data Storage");
    }
    
    // if local, move the item to the server
    if (true == recv_local) {
      // set local data by moving it, clean serialized data
      assert (data != nullptr);
      std::get<1>(*data) = std::vector<char>{};
      std::get<2>(*data) = std::make_shared<T>(std::forward<T>(item));
      // set serialize function
      std::get<3>(*data) = internal::serialize_any_ptr<T>;
      // free data->first that must be gotten with test_and_set()) from the server;
      std::get<0>(*data).clear();
      COUT << "zmq_data_service::set recv shared_data_ptr atomic clear" << ENDL;
    }
    // unlock access to this socket
    servers_.at(send_id).second.second.unlock();
    return ref_out;
  }

private:
    //using shared_data_type = std::pair<std::atomic_flag,boost::any>;
    using shared_data_type = std::tuple<std::atomic_flag,
                                    std::vector<char>,
                                    boost::any,
                                    std::function<std::vector<char>(boost::any)>>;

    /// default local machine
    //constexpr static char default_machine_[] = "localhost";
    /// default machine id
    constexpr static long default_id_ = 0;
    /// default number of tokens
    constexpr static long default_numTokens_ = 100;
    /// default data server port
    constexpr static long default_dataServer_port_ = 0;

    /// intra-process pattern
    const std::vector<std::string> inprocPattern {"inproc://zmq_data_service_", ""};
    /// inter-process pattern
    const std::vector<std::string> ipcPattern {"ipc:///tmp/zmq_data_service_", ".ipc"};
    /// tcp bind pattern
    const std::vector<std::string> tcpBindPattern {"tcp://*:0", ""};
    /// tcp connect pattern
    const std::vector<std::string> tcpConnectPattern {"tcp://", ":"};


    /// tag for set command
    const std::string setCmd{"SET"};
    /// tag for get command
    const std::string getCmd{"GET"};
    /// tag for end command
    const std::string endCmd{"END"};


  std::map<long, std::string> machines_;
  long id_;
  long dataServer_port_ = 0;
  long numTokens_ = 100;
  std::map<std::string, long> machinesMap_;
  // list of socket / (local_flag / mutex) for each and all servers
  std::map<long, std::pair<zmq::socket_t, std::pair<bool, std::mutex>>> servers_;
  std::vector<std::pair<bool,zmq::socket_t>> localServer_;
  long numLocalServers_ = 0;
  zmq::context_t context_;
  bool isServer_ = false;
  long server_id_ = -1;
  /// Proxy server address
  std::thread server_thread_;
  // data vector (used_flag, clean_counter,
  //              [atomic_flag, serial_data, local_data, serialize_func]) )
  std::vector<std::tuple<bool, long,
                        std::unique_ptr<shared_data_type>>> data_;
  boost::circular_buffer<long> emptyList_;
  std::shared_ptr<zmq_port_service> port_service_;

  /**
  \brief init_data_service function to init the data service.
  */
  void init_data_service ()
  {
    COUT << "zmq_data_service::init_data_service BEGIN" << ENDL;
    try {
    // set data array
    data_.resize(numTokens_);
    // init atomic flags to clear
    for (auto it=data_.begin(); it!=data_.end(); it++) {
      std::get<2>(*it) = std::make_unique<shared_data_type>();
      std::get<1>(*(std::get<2>(*it))).clear();
    }

    // init empty list
    emptyList_.set_capacity(numTokens_);
    for (long i=0; i<numTokens_; i++) {
       emptyList_.push_back(i);
    }
    COUT << "zmq_data_service::init_data_service init done" << ENDL;

    // set zmq context
    context_ = zmq::context_t(1);

    bool isRequiredIPC = false;
    bool isRequiredTCP = false;
    // set map of unique machines
    //for (long i=0; i<(long)machines_.size(); i++) {
    for (const auto& elem : machines_) {
      try {
      long i = elem.first;
      if (0 == machinesMap_.count(machines_[i])) {
        machinesMap_[machines_[i]] = i;
        if (i == id_) {
          isServer_ = true; // set true if is a server
          server_id_ = id_;
          COUT << "zmq_data_service::init_data_service current process: " << i << ENDL;
        } else if (machines_[i] == machines_[id_]) {
          server_id_ = i;
          COUT << "zmq_data_service::init_data_service same machine than current: " << i << ENDL;
        } else {
          isRequiredTCP = true;
          COUT << "zmq_data_service::init_data_service remote machine 1: " << i << ENDL;
        }
      } else if (machines_[i] == machines_[id_]) {
          isRequiredIPC = true;
          COUT << "zmq_data_service::init_data_service: remote machine 2" << i << ENDL;
      }
      } catch(const std::exception &e) {
        std::cerr << "zmq_data_service:init_data_service - for (const auto& elem : machines_) {} " << e.what() << std::endl;
      }
    }
    for (long i=0; i<(long)machines_.size(); i++) {
        COUT << "zmq_data_service::init_data_service machines_[" << i << "]: " << machines_[i] << ENDL;
    }
    for (auto it=machinesMap_.begin(); it!=machinesMap_.end(); it++) {
        COUT << "zmq_data_service::init_data_service machinesMap_[" << it->first << "]: " << it->second << ENDL;
    }
    COUT << "zmq_data_service::init_data_service isRequiredIPC: " << isRequiredIPC << ", isRequiredTCP: " << isRequiredTCP << ", isServer_: " << isServer_ << ENDL;

    // set the server socket and thread (if needed)
    if (isServer_) {
      try {
        long pos=0;
        for (long i=0; i<3; i++) {
          if (i == 0) {
            // inproc server socket binded
            std::ostringstream ss;
            ss << inprocPattern[0] << dataServer_port_ << inprocPattern[1];
            COUT << "zmq_data_service::init_data_service inproc bind url: " << ss.str() << ENDL;
            localServer_.emplace_back(std::piecewise_construct,
                                      std::forward_as_tuple(true),
                                      std::forward_as_tuple(context_,ZMQ_REP));
            localServer_[pos].second.bind(ss.str());
            numLocalServers_++;
            pos++;
            COUT << "zmq_data_service::init_data_service inproc bind end" << ENDL;
          } else if ( (i == 1) && (isRequiredIPC) ) {
            // ipc server socket binded
            try {
              std::ostringstream ss;
              ss << ipcPattern[0] << dataServer_port_ << ipcPattern[1];
              COUT << "zmq_data_service::init_data_service ipc bind url: " << ss.str() << ENDL;
              localServer_.emplace_back(std::piecewise_construct,
                                        std::forward_as_tuple(false),
                                        std::forward_as_tuple(context_,ZMQ_REP));
              localServer_[pos].second.bind(ss.str());
              numLocalServers_++;
              pos++;
              COUT << "zmq_data_service::init_data_service inproc bind end" << ENDL;
            } catch (std::exception& e) {
              // if not supported erase and enable TCP
              try {
                localServer_.at(pos);
                localServer_.erase(localServer_.begin() + pos);
              } catch (...){}
              isRequiredTCP = true;
            }
          } else if ( (i == 2) && (isRequiredTCP) ) {
            // tcp server socket binded
            std::ostringstream ss;
            ss << tcpBindPattern[0];
            COUT << "zmq_data_service::init_data_service tcp bind url: " << ss.str() << ENDL;
            localServer_.emplace_back(std::piecewise_construct,
                                      std::forward_as_tuple(false),
                                      std::forward_as_tuple(context_,ZMQ_REP));
            localServer_[pos].second.bind(ss.str());
            size_t size = 256;
            char buf[256];
            COUT << "zmq_data_service::init_data_service tcp bind getsockopt" << ENDL;
            localServer_[pos].second.getsockopt(ZMQ_LAST_ENDPOINT,buf,&size);
            std::string address(buf);
            std::string delimiter = ":";
            long pos = address.find(delimiter, address.find(delimiter)+1)+1;
            std::string srtPort = address.substr(pos); // token is "scott"
            COUT << "zmq_data_service::init_data_service tcp bind port" << srtPort << ENDL;

            long port = atoi(srtPort.c_str());
            COUT << "zmq_data_service::init_data_service tcp bind data: " << address << " (" << id_ << "," << dataServer_port_ << "," << port << ")" << ENDL;
            port_service_->set(id_,dataServer_port_,port);
            numLocalServers_++;
            pos++;
            COUT << "zmq_data_service::init_data_service tcp bind end" << ENDL;
          }
        }
        COUT << "zmq_data_service::init_data_service end binding sockets" << ENDL;

        // server thread launched
        server_thread_ = std::thread(&zmq_data_service::server_func, this);
        COUT << "zmq_data_service::init_data_service launched data server" << ENDL;
      } catch(const std::exception &e) {
        std::cerr << "zmq_data_service:init_data_service - if (isServer_){} " << e.what() << std::endl;
      }

    }
    COUT << "zmq_data_service::init_data_service connect sockets begin" << ENDL;

    // set the vector of client sockets
    for (const auto& elem : machinesMap_) {
      try {
      servers_.emplace(std::piecewise_construct,
                       std::forward_as_tuple(elem.second),
                       std::forward_as_tuple(
                            std::piecewise_construct,
                            std::forward_as_tuple(context_,ZMQ_REQ),
                            std::forward_as_tuple(
                                 std::piecewise_construct,
                                 std::forward_as_tuple(false),
                                 std::forward_as_tuple() ) ) );
      if (elem.second == id_) { /* same machine and process */
        std::ostringstream ss;
        ss << inprocPattern[0] << dataServer_port_ << inprocPattern[1];
        COUT << "zmq_data_service::init_data_service inproc connect: " << elem.second << ": " << ss.str() << ENDL;
        //connect and set local flag -> true
        servers_.at(elem.second).first.connect(ss.str());
        servers_.at(elem.second).second.first = true;
      } else if (elem.first == machines_[id_]) {
        try {
          std::ostringstream ss;
          ss << ipcPattern[0] << dataServer_port_ << ipcPattern[1];
          COUT << "zmq_data_service::init_data_service ipc connect: " << elem.second << ": " << ss.str() << ENDL;
          //connect and set local flag -> false
          servers_.at(elem.second).first.connect(ss.str());
          servers_.at(elem.second).second.first = false;
        } catch (std::exception& e) {  // if ipc sockets are not supported, change to TCP
          long port = port_service_->get(elem.second,dataServer_port_,true);
          COUT << "zmq_data_service::init_data_service tcp connect 1: (" << elem.second << "," << dataServer_port_ << "," << port << ")" << ENDL;
          std::ostringstream ss;
          ss << tcpConnectPattern[0] << elem.first << tcpConnectPattern[1] << port;
          COUT << "zmq_data_service::init_data_service tcp connect 1 data: " << ss.str() << ENDL;
          //connect and set local flag -> false
          servers_.at(elem.second).first.connect(ss.str());
          servers_.at(elem.second).second.first = false;
        }
      } else {
        long port = port_service_->get(elem.second,dataServer_port_,true);
        COUT << "zmq_data_servic::init_data_service tcp connect 2: (" << elem.second << "," << dataServer_port_ << "," << port << ")" << ENDL;
        std::ostringstream ss;
        ss << tcpConnectPattern[0] << elem.first << tcpConnectPattern[1] << port;
        COUT << "zmq_data_service::init_data_service tcp connect 2: " << ss.str() << ENDL;
        //connect and set local flag -> false
        servers_.at(elem.second).first.connect(ss.str());
        servers_.at(elem.second).second.first = false;
      }
      } catch(const std::exception &e) {
      std::cerr << "zmq_data_service:init_data_service - for (const auto& elem : machinesMap_) {}" << e.what() << std::endl;
      }
    }
    COUT << "zmq_data_service::init_data_service END" << ENDL;
    } catch(const std::exception &e) {
      std::cerr << "zmq_data_service:init_data_service" << e.what() << std::endl;
    }
  }

  /**
  \brief Server function to store and release data form the storage array.
  */
  void server_func ()
  {
    
    COUT << "zmq_data_service::server_func begin" << ENDL;
    //  Initialize poll set
    std::vector<zmq::pollitem_t> items;
    items.resize(numLocalServers_);
    for (long i=0; i<numLocalServers_; i++) {
      items[i] = zmq::pollitem_t{(void *)(localServer_[i].second), 0, ZMQ_POLLIN, 0 };
      //items[i].socket = (void *)(localServer_[i].second);
      //items[i].fd = 0;
      //items[i].events = ZMQ_POLLIN;
      //items[i].revents = 0;
      COUT << "zmq_data_service::server_func poll: " << i << ENDL;
    };

    while (1) {
      long next=0;
      
      COUT << "zmq_data_service::server_func waitting for polling" << ENDL;
      //wait until next connection
      //zmq::poll (&items [0], numLocalServers_, -1);
      zmq::poll (items.data(), numLocalServers_, -1);
      for (long i=0; i<numLocalServers_; i++) {
        if (items [i].revents & ZMQ_POLLIN) {
            next=i;
            COUT << "zmq_data_service::server_func wakeup socket: " << next << ", recv_local = " << localServer_[next].first << ENDL;
            break;
        }
      }
      
      // recv command
      zmq::message_t msg;
      localServer_[next].second.recv(&msg);

      // set command
      if ( (msg.size() == setCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(setCmd.data()),setCmd.size())) ) {
        COUT << "zmq_data_service::server_func SET" << ENDL;
        
        // get data reference even if it is null
        data_ref_type old_ref{};
        old_ref.recv(localServer_[next].second);
        
        // check old ref or get next empty slot
        long slot = -1; // ERROR by default;
        if ( (old_ref != data_ref_type{}) &&
             (true == std::get<0>(data_[old_ref.get_pos()])) ) {
            slot = old_ref.get_pos();
        } else if (false == emptyList_.empty()) {
          slot = emptyList_.front();
          emptyList_.pop_front();
          COUT << "zmq_data_service::server_func recv data: size = " << msg.size() << ENDL;
        }
        
        // receive local_flag
        localServer_[next].second.recv(&msg);
        bool recv_local = *((bool*) msg.data());
        assert (localServer_[next].first == recv_local);
        COUT << "zmq_data_service::server_func recv_local=" << recv_local << ENDL;

        // receive serialize data if not local
        if (false == recv_local) {
          // recv item and copy it back
          localServer_[next].second.recv(&msg);
          COUT << "zmq_data_service::server_func recv serialized item size=" << msg.size() << ENDL;
        }
        
        COUT << "zmq_data_service::server_func recv get slot=" << slot << ENDL;

        // fill slot, create data_ref_type, and data pointer
        data_ref_type ref{};
        shared_data_type *shared_data_ptr = nullptr;
        if (slot != -1) {
          ref = data_ref_type{id_,slot};
          std::get<0>(data_[slot])=true;
          std::get<1>(data_[slot])=0;
          // get data pointer
          shared_data_ptr = std::get<2>(data_[slot]).get();
          // store data if remote
          if (false == recv_local) {
            while(std::get<0>(*shared_data_ptr).test_and_set());
            // copy serialized data
            std::get<1>(*shared_data_ptr).resize(msg.size());
            std::memcpy(std::get<1>(*shared_data_ptr).data(), msg.data(), msg.size());
            // clean local data and serialize function
            std::get<2>(*shared_data_ptr) = boost::any{};
            std::get<3>(*shared_data_ptr) = nullptr;
            COUT << "zmq_data_service::server_func data stored: size = " << std::get<1>(*shared_data_ptr).size() << " slot = " << slot << ENDL;
            std::get<0>(*shared_data_ptr).clear();
          }
        } else {
          COUT << "zmq_data_service::server_func data NOT stored (FULL STORAGE)" << ENDL;
        }
        
        COUT << "zmq_data_service::server_func send set response" << ENDL;

        // serialize ref nd send back
        ref.send(localServer_[next].second, ZMQ_SNDMORE);
        COUT << "zmq_data_service::server_func send ref=(" << ref.get_id() << ", " << ref.get_pos() << ")" << ENDL;
        
        if (true == recv_local) {
          // send local_flag
          localServer_[next].second.send((bool *) &recv_local, sizeof(recv_local),ZMQ_SNDMORE);
          COUT << "zmq_data_service::server_func send recv_local=" << recv_local << ENDL;
          if (slot == -1) {
            // if no entry free send nullptr else
            shared_data_ptr = nullptr;
            localServer_[next].second.send((shared_data_type **) &shared_data_ptr, sizeof(shared_data_type *));
          } else {
            // wait until collect the atomic flag
            while(std::get<0>(*shared_data_ptr).test_and_set());
            // send shared data pointer
            localServer_[next].second.send((shared_data_type **) &shared_data_ptr, sizeof(shared_data_type *));
          }
        COUT << "zmq_data_service::server_func send shared_data_ptr=" << shared_data_ptr << ENDL;
        } else {
          // send local_flag
          localServer_[next].second.send((bool *) &recv_local, sizeof(recv_local));
          COUT << "zmq_data_service::server_func send recv_local=" << recv_local << ENDL;
        }
        COUT << "zmq_data_service::server_func send ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << ENDL;

      } else if ( (msg.size() == getCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(getCmd.data()), getCmd.size())) ) {
        COUT << "zmq_data_service::server_func GET" << ENDL;
        
        // recv and deserialized reference
        data_ref_type ref{};
        ref.recv(localServer_[next].second);
        COUT << "zmq_data_service::server_func decode ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << ENDL;

        // recv release flag and release count
        localServer_[next].second.recv(&msg);
        bool release_flag = *((bool*) msg.data());
        localServer_[next].second.recv(&msg);
        long release_count = *((long*) msg.data());

        COUT << "zmq_data_service::server_func release_flag = " << release_flag << ",  release_count = " << release_count << ENDL;

        //get reference slot, send it and set it as empty
        if ( (id_ == ref.get_id()) && (0 <= ref.get_pos()) &&
             ((long)data_.size() > ref.get_pos()) && std::get<0>(data_[ref.get_pos()])) {

          // check if is required to release the data position because the count is complete
          bool erase_flag = false;
          if (release_flag) {
            // increase the count
            std::get<1>(data_[ref.get_pos()])++;
            COUT << "zmq_data_service::server_func send data: release count increased: slot = " << ref.get_pos() << ", count=(" << std::get<1>(data_[ref.get_pos()]) << ", " << release_count << ")" << ENDL;
            if (std::get<1>(data_[ref.get_pos()]) >= release_count) {
              erase_flag = true;
            }
          }
          // send erase flag
          localServer_[next].second.send((bool *) &erase_flag, sizeof(erase_flag),ZMQ_SNDMORE);

          // send local_flag
          bool recv_local = localServer_[next].first;
          localServer_[next].second.send((bool *) &recv_local, sizeof(recv_local),ZMQ_SNDMORE);
          COUT << "zmq_data_service::server_func send recv_local = " << recv_local << ", localServer_[" << next << "].first = " << localServer_[next].first << ENDL;

          // get shared data pointer
          shared_data_type *shared_data_ptr = std::get<2>(data_[ref.get_pos()]).get();

          // check if local or remnote
          if (true == recv_local) {
            // wait until collect the atomic flag
            while(std::get<0>(*shared_data_ptr).test_and_set());
            // send shared data pointer
            localServer_[next].second.send((shared_data_type **) &shared_data_ptr, sizeof(shared_data_type *));
            COUT << "zmq_data_service::server_func send data: shared_data_ptr = " << shared_data_ptr << ENDL;
          } else {
            // wait until collect the atomic flag
            while(std::get<0>(*shared_data_ptr).test_and_set());
            // if seial string is empty, serialize the local object
            if (std::get<1>(*shared_data_ptr).size() == 0) {
              std::get<1>(*shared_data_ptr) = std::get<3>(*shared_data_ptr)(std::get<2>(*shared_data_ptr));
            }
            // send remote data
            localServer_[next].second.send(std::get<1>(*shared_data_ptr).data(),
                                           std::get<1>(*shared_data_ptr).size());
            COUT << "zmq_data_service::server_func send data: size = " << std   ::get<1>(*shared_data_ptr).size() << " slot = " << ref.get_pos() << ENDL;
            // atomic flag
            std::get<0>(*shared_data_ptr).clear();

          }
          
          // if requiered relase the slot
          if (true == erase_flag) {
            // count is complete -> release
            std::get<0>(data_[ref.get_pos()]) = false;
            std::get<1>(data_[ref.get_pos()]) = 0;
            // wait until collect the atomic flag
            while(std::get<0>(*shared_data_ptr).test_and_set());
            std::get<1>(*shared_data_ptr) = std::vector<char>{};
            std::get<2>(*shared_data_ptr) = boost::any{};
            std::get<3>(*shared_data_ptr) = nullptr;
            // clear atomic flag
            std::get<0>(*shared_data_ptr).clear();
            // put entry on empty list
            emptyList_.push_back(ref.get_pos());
            COUT << "zmq_data_service::server_func send data: slot released: slot = " << ref.get_pos() << ENDL;
          }
        } else {
          localServer_[next].second.send("", 0);
          COUT << "zmq_data_service::server_func: ERROR ref: (" << ref.get_id() << "," << ref.get_pos() << "), " << "server id: " << id_ << "data range: (" << 0 << "," << data_.size() << ")" << "data_slot_occupied = " << std::get<0>(data_[ref.get_pos()]) << ENDL;
        }
      } else if ( (msg.size() == endCmd.size()) &&
           (0 == std::memcmp(msg.data(), static_cast<const void*>(endCmd.data()), endCmd.size())) ) {
           COUT << "zmq_data_service::server_func END" << ENDL;
        break;
      }
    }
    // need to release sockets???
    COUT << "zmq_data_service::server_func end" << ENDL;
  }
};

/**
@}
*/


}

#endif
