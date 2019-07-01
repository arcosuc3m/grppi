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

#include <zmq.hpp>

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/circular_buffer.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
//#pragma GCC diagnostic pop

#include "zmq_port_service.h"
#include "zmq_data_reference.h"

namespace grppi{

/**
\defgroup data_service zmq_data_service
\brief Data service support types.
@{
*/

class zmq_data_service {
public:
   
  // no copy constructors
  zmq_data_service(const zmq_data_service&) =delete;
  zmq_data_service& operator=(zmq_data_service const&) =delete;

  /**
  \brief Data Service contructor interface. Also creates one data server per node.
  \param dataServer_port common port for port_service and local comms.
  \param numTokens num. of maximum elements to store
  */
  zmq_data_service(int dataServer_port, int numTokens) :
    //machines_(1,default_machine_),
    machines_(1,"localhost"),
    id_(default_id_),
    dataServer_port_(dataServer_port),
    numTokens_(numTokens),
    port_service_()
  {
      std::cout << "zmq_data_service: " << machines_[0] << std::endl;
      init_data_service ();
  }
  
  zmq_data_service(int dataServer_port) :
    zmq_data_service (dataServer_port, default_numTokens_) {}

  zmq_data_service() :
    zmq_data_service (default_dataServer_port_, default_numTokens_) {}
  
  /**
  \brief Data Service contructor interface. Also creates one data server per node.
  \param machines list of machine nodes.
  \param id position of this node on the machine list.
  \param port_service service to store and retrive comunication ports.
  \param dataServer_port common port for port_service and local comms.
  \param numTokens num. of maximum elements to store
  */
  zmq_data_service(std::vector<std::string> machines, int id,
              std::shared_ptr<zmq_port_service> port_service,
              int dataServer_port, int numTokens) :
    machines_(machines.begin(), machines.end()),
    id_(id),
    dataServer_port_(dataServer_port),
    numTokens_(numTokens),
    port_service_(port_service)
  {
    init_data_service ();
  }

  zmq_data_service(std::vector<std::string> machines, int id,
              std::shared_ptr<zmq_port_service> port_service, int dataServer_port) :
    zmq_data_service (machines, id, port_service, dataServer_port, default_numTokens_) {}

  zmq_data_service(std::vector<std::string> machines, int id,
              std::shared_ptr<zmq_port_service> port_service) :
    zmq_data_service (machines, id, port_service, default_dataServer_port_, default_numTokens_) {}

  /**
  \brief Data Server destructor. Also finish the data server if exists.
  \param machines list of machine nodes.
  \param id position of this node on the machine list.
  \param numTokens num. of maximum elements to store
  */
  ~zmq_data_service()
  {
    std::cout << "~zmq_data_service begin" << std::endl;
    if (isServer_) {
        std::cout << "join proxy_thread_" << std::endl;
        servers_.at(id_).send(endCmd.data(), endCmd.size(), 0);
        server_thread_.join();
    }
    servers_.clear();
    localServer_.clear();
    std::cout << "~zmq_data_service end" << std::endl;
  }

  /**
  \brief Get the data element from the server and position
  referenced in the ref param.
  \tparam T Element type for the data element.
  \param ref zmq_data_reference of the server and position for tha data.
  */
  template <class T>
  T get (zmq_data_reference ref)
  {
    mutex.lock();
    int send_id = ref.get_id();
    std::cout << "get send id:" << send_id << std::endl;

    // send the command tag
    servers_.at(send_id).send(getCmd.data(), getCmd.size(), ZMQ_SNDMORE);
    std::cout << "get send cmd GET" << std::endl;

    
    // serialize obj into an std::string
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
    boost::archive::binary_oarchive oa(os);

    oa << ref;
    os.flush();
    
    // send the reference (server_id,pos)
    servers_.at(send_id).send(serial_str.data(), serial_str.length());
    std::cout << "get send ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;

    // receive the data
    zmq::message_t message;
    servers_.at(send_id).recv(&message);
    std::cout << "get rec data: size=" << message.size() << std::endl;

    if (message.size() == 0) {
        std::cout << "Error Item not found" << std::endl;
        throw std::runtime_error("Item not found");
    }
    
    // wrap buffer inside a stream and deserialize serial_str into obj
    boost::iostreams::basic_array_source<char> device((char *)message.data(), message.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > is(device);
    boost::archive::binary_iarchive ia(is);

    T item;
    try {
      ia >> item;
    } catch (...) {
        throw std::runtime_error("Incorrect Type");
    }

    mutex.unlock();
    return item;
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
    mutex.lock();
    std::cout << "set send id:" << server_id_ << std::endl;
    // send the command tag
    servers_.at(server_id_).send(setCmd.data(), setCmd.size(), ZMQ_SNDMORE);
    std::cout << "set send cmd SET" << std::endl;

    // serialize obj into an std::string
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
    boost::archive::binary_oarchive oa(os);

    try {
        oa << item;
        os.flush();
    } catch (...) {
        throw std::runtime_error("Type not serializable");
    }

    // send the data
    servers_.at(server_id_).send(serial_str.data(), serial_str.length());
    std::cout << "set send data: size=" << serial_str.length() << std::endl;

    // receive the reference (server_id,pos)
    zmq::message_t message;
    servers_.at(server_id_).recv(&message);

    // wrap buffer inside a stream and deserialize serial_str into obj
    boost::iostreams::basic_array_source<char> device((char *)message.data(), message.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > is(device);
    boost::archive::binary_iarchive ia(is);

    zmq_data_reference ref;
    ia >> ref;
    std::cout << "set recv ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
    if (ref.get_pos() == -1) {
        std::cout << "Error full data storage" << std::endl;
        throw std::runtime_error("Full Data Storage");
    }
    mutex.unlock();
    return ref;
  }

private:

    /// default local machine
    //constexpr static char default_machine_[] = "localhost";
    /// default machine id
    constexpr static int default_id_ = 0;
    /// default number of tokens
    constexpr static int default_numTokens_ = 100;
    /// default data server port
    constexpr static int default_dataServer_port_ = 0;

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


  std::vector<std::string> machines_;
  int id_;
  int dataServer_port_ = 0;
  int numTokens_ = 100;
  std::map<std::string, int> machinesMap_;
  std::map<int, zmq::socket_t> servers_;
  std::vector<zmq::socket_t> localServer_;
  int numLocalServers_ = 0;
  zmq::context_t context_;
  bool isServer_ = false;
  int server_id_ = -1;
  /// Proxy server address
  std::thread server_thread_;
  std::vector<std::pair<bool,std::vector<unsigned char>>> data_;
  boost::circular_buffer<int> emptyList_;
  std::shared_ptr<zmq_port_service> port_service_;
  // lock for get and set operations
  std::mutex mutex;

  /**
  \brief init_data_service function to init the data service.
  */
  void init_data_service ()
  {
    std::cout << "zmq_data_service 1" << std::endl;
    
    // set data array
    data_.resize(numTokens_);

    // init empty list
    emptyList_.set_capacity(numTokens_);
    for (int i=0; i<numTokens_; i++) {
       emptyList_.push_back(i);
    }
    std::cout << "zmq_data_service 2" << std::endl;

    // set zmq context
    context_ = zmq::context_t(1);

    bool isRequiredIPC = false;
    bool isRequiredTCP = false;
    // set map of unique machines
    for (int i=0; i<(int)machines_.size(); i++) {
      if (0 == machinesMap_.count(machines_[i])) {
        machinesMap_[machines_[i]] = i;
        if (i == id_) {
          isServer_ = true; // set true if is a server
          server_id_ = id_;
          std::cout << "uno: " << i << std::endl;
        } else if (machines_[i] == machines_[id_]) {
          server_id_ = i;
          std::cout << "dos: " << i << std::endl;
        } else {
          isRequiredTCP = true;
          std::cout << "tres: " << i << std::endl;
        }
      } else if (machines_[i] == machines_[id_]) {
          isRequiredIPC = true;
          std::cout << "cuatro: " << i << std::endl;
      }
    }
    for (int i=0; i<(int)machines_.size(); i++) {
        std::cout << "machines_[" << i << "]: " << machines_[i] << std::endl;
    }
    for (auto it=machinesMap_.begin(); it!=machinesMap_.end(); it++) {
        std::cout << "machinesMap_[" << it->first << "]: " << it->second << std::endl;
    }
    std::cout << "isRequiredIPC: " << isRequiredIPC << std::endl;
    std::cout << "isRequiredTCP: " << isRequiredTCP << std::endl;
    std::cout << "isServer_: " << isServer_ << std::endl;
    std::cout << "zmq_data_service 3" << std::endl;

    // set the server socket and thread (if needed)
    if (isServer_) {
        int pos=0;
        for (int i=0; i<3; i++) {
          if (i == 0) {
            // inproc server socket binded
            std::ostringstream ss;
            ss << inprocPattern[0] << dataServer_port_ << inprocPattern[1];
            std::cout << "zmq_data_service 3.1: " << ss.str() << std::endl;
            localServer_.emplace_back(context_,ZMQ_REP);
            localServer_[pos].bind(ss.str());
            numLocalServers_++;
            pos++;
            std::cout << "zmq_data_service 3.1 end" << std::endl;
          } else if ( (i == 1) && (isRequiredIPC) ) {
            // ipc server socket binded
            try {
              std::ostringstream ss;
              ss << ipcPattern[0] << dataServer_port_ << ipcPattern[1];
              std::cout << "zmq_data_service 3.2: " << ss.str() << std::endl;
              localServer_.emplace_back(context_,ZMQ_REP);
              localServer_[pos].bind(ss.str());
              numLocalServers_++;
              pos++;
              std::cout << "zmq_data_service 3.2 end" << std::endl;
            } catch (std::exception& e) {
              // if not supported erase and enable TCP
              try {
                localServer_.at(pos);
                localServer_.erase(localServer_.begin() + pos);
              } catch (...){}
              isRequiredTCP = true;
            }
          } else if ( (i == 2) && (isRequiredTCP) ) {
            // inproc server socket binded
            std::ostringstream ss;
            ss << tcpBindPattern[0];
            std::cout << "zmq_data_service 3.3: " << ss.str() << std::endl;
            localServer_.emplace_back(context_,ZMQ_REP);
            localServer_[pos].bind(ss.str());
            size_t size = 256;
            char buf[256];
            std::cout << "zmq_data_service 3.3: getsockopt" << std::endl;
            localServer_[pos].getsockopt(ZMQ_LAST_ENDPOINT,buf,&size);
            std::string address(buf);
            std::string delimiter = ":";
            int pos = address.find(delimiter, address.find(delimiter)+1)+1;
            std::string srtPort = address.substr(pos); // token is "scott"
            std::cout << "zmq_data_service 3.3: " << srtPort << std::endl;

            int port = atoi(srtPort.c_str());
            std::cout << "zmq_data_service 3.3: " << address << " (" << id_ << "," << dataServer_port_ << "," << port << ")" << std::endl;
            port_service_->set(id_,dataServer_port_,port);
            numLocalServers_++;
            pos++;
            std::cout << "zmq_data_service 3.3 end" << std::endl;
          }
        }
        std::cout << "zmq_data_service 4" << std::endl;

        // server thread launched
        server_thread_ = std::thread(&zmq_data_service::server_func, this);
        std::cout << "zmq_data_service 5" << std::endl;

    }
    std::cout << "zmq_data_service 6" << std::endl;

    // set the vector of client sockets
    for (const auto& elem : machinesMap_) {
      servers_.emplace(std::piecewise_construct,
                       std::forward_as_tuple(elem.second),
                       std::forward_as_tuple(context_,ZMQ_REQ));
      if (elem.second == id_) { /* same machine and process */
        std::ostringstream ss;
        ss << inprocPattern[0] << dataServer_port_ << inprocPattern[1];
        std::cout << "zmq_data_service 6.3: " << elem.second << ": " << ss.str() << std::endl;
        servers_.at(elem.second).connect(ss.str());
      } else if (elem.first == machines_[id_]) {
        try {
          std::ostringstream ss;
          ss << ipcPattern[0] << dataServer_port_ << ipcPattern[1];
          std::cout << "zmq_data_service 6.3: " << elem.second << ": " << ss.str() << std::endl;
          servers_.at(elem.second).connect(ss.str());
        } catch (std::exception& e) {  // if ipc sockets are not supported, change to TCP
          int port = port_service_->get(elem.second,dataServer_port_,true);
          std::cout << "zmq_data_service 6.3: (" << elem.second << "," << dataServer_port_ << "," << port << ")" << std::endl;
          std::ostringstream ss;
          ss << tcpConnectPattern[0] << elem.first << tcpConnectPattern[1] << port;
          std::cout << "zmq_data_service 6.3: " << ss.str() << std::endl;
          servers_.at(elem.second).connect(ss.str());
        }
      } else {
        int port = port_service_->get(elem.second,dataServer_port_,true);
        std::cout << "zmq_data_service 6.3: (" << elem.second << "," << dataServer_port_ << "," << port << ")" << std::endl;
        std::ostringstream ss;
        ss << tcpConnectPattern[0] << elem.first << tcpConnectPattern[1] << port;
        std::cout << "zmq_data_service 6.3: " << ss.str() << std::endl;
        servers_.at(elem.second).connect(ss.str());
      }
    }
    std::cout << "zmq_data_service end" << std::endl;
  }

  /**
  \brief Server function to store and release data form the storage array.
  */
  void server_func ()
  {
    
    std::cout << "server_func begin" << std::endl;
    //  Initialize poll set
    std::vector<zmq::pollitem_t> items;
    items.resize(numLocalServers_);
    for (int i=0; i<numLocalServers_; i++) {
      items[i] = zmq::pollitem_t{localServer_[i], 0, ZMQ_POLLIN, 0 };
      std::cout << "server_func poll: " << i << std::endl;
    };

    while (1) {
      int next=0;
      
      //wait until next connection
      zmq::poll (&items [0], numLocalServers_, -1);
      for (int i=0; i<numLocalServers_; i++) {
        if (items [i].revents & ZMQ_POLLIN) {
            next=i;
            std::cout << "server_func wakeup socket: " << next << std::endl;
            break;
        }
      }
      
      // recv command
      zmq::message_t msg;
      localServer_[next].recv(&msg);

      // set command
      if ( (msg.size() == setCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(setCmd.data()),setCmd.size())) ) {
        std::cout << "server_func SET" << std::endl;
        
        // get next empty slot, recv item and copy it back
        int slot = -1; // ERROR by default;
        localServer_[next].recv(&msg);
        if (false == emptyList_.empty()) {
          slot = emptyList_.front();
          emptyList_.pop_front();
          std::cout << "server_func recv data: size = " << msg.size() << std::endl;
          data_[slot].first=true;
          data_[slot].second.resize(msg.size());
          std::memcpy(data_[slot].second.data(), msg.data(), msg.size());
          std::cout << "server_func data stored: size = " << data_[slot].second.size() << " slot = " << slot << std::endl;
        } else {
          std::cout << "server_func data NOT stored (FULL STORAGE)" << std::endl;
        }
        // create zmq_data_reference, serialize and send back
        zmq_data_reference ref(id_,slot);
        std::string serial_str;
        boost::iostreams::back_insert_device<std::string> inserter(serial_str);
        boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
        boost::archive::binary_oarchive oa(s);

        oa << ref;
        s.flush();

        localServer_[next].send(serial_str.data(), serial_str.length());
        std::cout << "server_func send ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;

      } else if ( (msg.size() == getCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(getCmd.data()), getCmd.size())) ) {
        std::cout << "server_func GET" << std::endl;
        
        // recv and deserialized reference
        localServer_[next].recv(&msg);
        std::cout << "server_func recv ref" << std::endl;

        boost::iostreams::basic_array_source<char> device((char *)msg.data(), msg.size());
        boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
        boost::archive::binary_iarchive ia(s);
  
        zmq_data_reference ref;
        ia >> ref;
        std::cout << "server_func decode ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;

        
        //get reference slot, send it and set it as empty
        if ( (id_ == ref.get_id()) && (0 <= ref.get_pos()) &&
             ((int)data_.size() > ref.get_pos()) && data_[ref.get_pos()].first) {
          localServer_[next].send(data_[ref.get_pos()].second.data(),
                                  data_[ref.get_pos()].second.size());
          data_[ref.get_pos()].first = false;
          emptyList_.push_back(ref.get_pos());
          std::cout << "server_func send data: size = " << data_[ref.get_pos()].second.size() << " slot = " << ref.get_pos() << std::endl;
        } else {
          localServer_[next].send("", 0);
          std::cout << "server_func: ERROR ref: (" << ref.get_id() << "," << ref.get_pos() << "), ";
          std::cout << "server id: " << id_ << "data range: (" << 0 << "," << data_.size() << ")";
          std::cout << "data_slot_occupied = " << data_[ref.get_pos()].first << std::endl;
        }
      } else if ( (msg.size() == endCmd.size()) &&
           (0 == std::memcmp(msg.data(), static_cast<const void*>(endCmd.data()), endCmd.size())) ) {
           std::cout << "server_func END" << std::endl;
        break;
      }
    }
    // need to release sockets???
    std::cout << "server_func end" << std::endl;
  }
};

/**
@}
*/


}

#endif
