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
        servers_.at(id_).send(endCmd.data(), endCmd.size(), 0);
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
    mutex.lock();
    long send_id = ref.get_id();
    COUT << "get send id:" << send_id << ENDL;

    // send the command tag
    servers_.at(send_id).send(getCmd.data(), getCmd.size(), ZMQ_SNDMORE);
    COUT << "get send cmd GET" << ENDL;

    
    // serialize obj into an std::string
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
    boost::archive::binary_oarchive oa(os);

    oa << ref;
    os.flush();
    
    // send the reference (server_id,pos), release flag and release count
    servers_.at(send_id).send(serial_str.data(), serial_str.length(), ZMQ_SNDMORE);
    servers_.at(send_id).send((bool *)(&release), sizeof(release), ZMQ_SNDMORE);
    servers_.at(send_id).send((long *)(&release_count), sizeof(release_count));
    COUT << "zmq_data_service::get send ref: (" << ref.get_id() << "," << ref.get_pos() << "), release: " << release << ", release_count: " << release_count << ENDL;

    // receive the data
    zmq::message_t message;
    servers_.at(send_id).recv(&message);
    COUT << "zmq_data_service::get rec data: size=" << message.size() << ENDL;

    if (message.size() == 0) {
        COUT << "zmq_data_service::get Error Item not found" << ENDL;
        mutex.unlock();
        throw std::runtime_error("zmq_data_service::get Item not found");
    }
    
    // wrap buffer inside a stream and deserialize serial_str into obj
    boost::iostreams::basic_array_source<char> device((char *)message.data(), message.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > is(device);
    boost::archive::binary_iarchive ia(is);

    T item;
    try {
      ia >> item;
    } catch (...) {
        mutex.unlock();
        throw std::runtime_error("zmq_data_service::get Incorrect Type");
    }

    mutex.unlock();
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
  data_ref_type set(T item, data_ref_type ref_in = data_ref_type{})
  {

    // if ref_in is valid  uses it server, if not uses local server
    long send_id = server_id_;
    if (ref_in != data_ref_type{}) {
        send_id = ref_in.get_id();
    }
    COUT << "zmq_data_service::set send id:" << send_id << ENDL;

    // serialize obj into an std::string
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
    boost::archive::binary_oarchive oa(os);

    try {
        oa << item;
        os.flush();
    } catch (...) {
        throw std::runtime_error("zmq_data_service::set Type not serializable");
    }

    // lock the access for other threads
    mutex.lock();

    // send the command tag
    servers_.at(send_id).send(setCmd.data(), setCmd.size(), ZMQ_SNDMORE);
    COUT << "zmq_data_service::set send cmd SET" << ENDL;

    // send the data reference even if it is null
    std::string ref_in_string = ref_in.get_serialized_string();
    servers_.at(send_id).send((char *)(ref_in_string.data()), ref_in_string.size(), ZMQ_SNDMORE);
    COUT << "zmq_data_service::set send data reference: size=" << ref_in_string.length() << ENDL;

    // send the data
    servers_.at(send_id).send(serial_str.data(), serial_str.length());
    COUT << "zmq_data_service::set send data: size=" << serial_str.length() << ENDL;

    // receive the reference (server_id,pos)
    zmq::message_t message;
    servers_.at(send_id).recv(&message);

    if (message.size() == 0) {
        COUT << "zmq_data_service::set ERROR: does not return a data ref" << ENDL;
        mutex.unlock();
        throw std::runtime_error("zmq_data_service::set Does not return a data ref");
    }

    data_ref_type ref_out{};
    ref_out.set_serialized_string((char *)message.data(),message.size());
    COUT << "zmq_data_service::set recv ref: (" << ref_out.get_id() << "," << ref_out.get_pos() << ")" << ENDL;

    if (ref_out == data_ref_type{}) {
        std::cerr << "zmq_data_service::set Error full data storage" << std::endl;
        mutex.unlock();
        throw std::runtime_error("zmq_data_service::set Full Data Storage");
    }
    mutex.unlock();
    return ref_out;
  }

private:

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
  std::map<long, zmq::socket_t> servers_;
  std::vector<zmq::socket_t> localServer_;
  long numLocalServers_ = 0;
  zmq::context_t context_;
  bool isServer_ = false;
  long server_id_ = -1;
  /// Proxy server address
  std::thread server_thread_;
  std::vector<std::tuple<bool, long, std::vector<unsigned char>>> data_;
  boost::circular_buffer<long> emptyList_;
  std::shared_ptr<zmq_port_service> port_service_;
  // lock for get and set operations
  std::mutex mutex;

  /**
  \brief init_data_service function to init the data service.
  */
  void init_data_service ()
  {
    COUT << "zmq_data_service::init_data_service BEGIN" << ENDL;
    try {
    // set data array
    data_.resize(numTokens_);

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
            localServer_.emplace_back(context_,ZMQ_REP);
            localServer_[pos].bind(ss.str());
            numLocalServers_++;
            pos++;
            COUT << "zmq_data_service::init_data_service inproc bind end" << ENDL;
          } else if ( (i == 1) && (isRequiredIPC) ) {
            // ipc server socket binded
            try {
              std::ostringstream ss;
              ss << ipcPattern[0] << dataServer_port_ << ipcPattern[1];
              COUT << "zmq_data_service::init_data_service ipc bind url: " << ss.str() << ENDL;
              localServer_.emplace_back(context_,ZMQ_REP);
              localServer_[pos].bind(ss.str());
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
            // inproc server socket binded
            std::ostringstream ss;
            ss << tcpBindPattern[0];
            COUT << "zmq_data_service::init_data_service tcp bind url: " << ss.str() << ENDL;
            localServer_.emplace_back(context_,ZMQ_REP);
            localServer_[pos].bind(ss.str());
            size_t size = 256;
            char buf[256];
            COUT << "zmq_data_service::init_data_service tcp bind getsockopt" << ENDL;
            localServer_[pos].getsockopt(ZMQ_LAST_ENDPOINT,buf,&size);
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
                       std::forward_as_tuple(context_,ZMQ_REQ));
      if (elem.second == id_) { /* same machine and process */
        std::ostringstream ss;
        ss << inprocPattern[0] << dataServer_port_ << inprocPattern[1];
        COUT << "zmq_data_service::init_data_service inproc connect: " << elem.second << ": " << ss.str() << ENDL;
        servers_.at(elem.second).connect(ss.str());
      } else if (elem.first == machines_[id_]) {
        try {
          std::ostringstream ss;
          ss << ipcPattern[0] << dataServer_port_ << ipcPattern[1];
          COUT << "zmq_data_service::init_data_service ipc connect: " << elem.second << ": " << ss.str() << ENDL;
          servers_.at(elem.second).connect(ss.str());
        } catch (std::exception& e) {  // if ipc sockets are not supported, change to TCP
          long port = port_service_->get(elem.second,dataServer_port_,true);
          COUT << "zmq_data_service::init_data_service tcp connect 1: (" << elem.second << "," << dataServer_port_ << "," << port << ")" << ENDL;
          std::ostringstream ss;
          ss << tcpConnectPattern[0] << elem.first << tcpConnectPattern[1] << port;
          COUT << "zmq_data_service::init_data_service tcp connect 1 data: " << ss.str() << ENDL;
          servers_.at(elem.second).connect(ss.str());
        }
      } else {
        long port = port_service_->get(elem.second,dataServer_port_,true);
        COUT << "zmq_data_servic::init_data_service tcp connect 2: (" << elem.second << "," << dataServer_port_ << "," << port << ")" << ENDL;
        std::ostringstream ss;
        ss << tcpConnectPattern[0] << elem.first << tcpConnectPattern[1] << port;
        COUT << "zmq_data_service::init_data_service tcp connect 2: " << ss.str() << ENDL;
        servers_.at(elem.second).connect(ss.str());
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
      items[i] = zmq::pollitem_t{(void *)localServer_[i], 0, ZMQ_POLLIN, 0 };
      //items[i].socket = (void *)localServer_[i];
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
            COUT << "zmq_data_service::server_func wakeup socket: " << next << ENDL;
            break;
        }
      }
      
      // recv command
      zmq::message_t msg;
      localServer_[next].recv(&msg);

      // set command
      if ( (msg.size() == setCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(setCmd.data()),setCmd.size())) ) {
        COUT << "zmq_data_service::server_func SET" << ENDL;
        
        // get data reference even if it is null
        localServer_[next].recv(&msg);
        data_ref_type old_ref{};
        old_ref.set_serialized_string((char *)msg.data(),msg.size());
        assert (old_ref != data_ref_type{});  //??? TEMP
        
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
        // recv item and copy it back
        localServer_[next].recv(&msg);
        if (slot != -1) {
          std::get<0>(data_[slot])=true;
          std::get<1>(data_[slot])=0;
          std::get<2>(data_[slot]).resize(msg.size());
          std::memcpy(std::get<2>(data_[slot]).data(), msg.data(), msg.size());
          COUT << "zmq_data_service::server_func data stored: size = " << std::get<2>(data_[slot]).size() << " slot = " << slot << ENDL;
        } else {
          COUT << "zmq_data_service::server_func data NOT stored (FULL STORAGE)" << ENDL;
        }
        // create data_ref_type, serialize and send back
        data_ref_type ref{};
        if (slot != -1) {
            ref = data_ref_type{id_,slot};
        }
        std::string serial_str;
        boost::iostreams::back_insert_device<std::string> inserter(serial_str);
        boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
        boost::archive::binary_oarchive oa(s);

        oa << ref;
        s.flush();

        localServer_[next].send(serial_str.data(), serial_str.length());
        COUT << "zmq_data_service::server_func send ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << ENDL;

      } else if ( (msg.size() == getCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(getCmd.data()), getCmd.size())) ) {
        COUT << "zmq_data_service::server_func GET" << ENDL;
        
        // recv and deserialized reference
        localServer_[next].recv(&msg);
        COUT << "zmq_data_service::server_func recv ref" << ENDL;
        data_ref_type ref{};
        ref.set_serialized_string((char *)msg.data(),msg.size());
        COUT << "zmq_data_service::server_func decode ref: (" << ref.get_id() << "," << ref.get_pos() << ")" << ENDL;

        // recv release flag and release count
        localServer_[next].recv(&msg);
        bool release_flag = *((bool*) msg.data());
        localServer_[next].recv(&msg);
        long release_count = *((long*) msg.data());
        
        //get reference slot, send it and set it as empty
        if ( (id_ == ref.get_id()) && (0 <= ref.get_pos()) &&
             ((long)data_.size() > ref.get_pos()) && std::get<0>(data_[ref.get_pos()])) {
          localServer_[next].send(std::get<2>(data_[ref.get_pos()]).data(),
                                  std::get<2>(data_[ref.get_pos()]).size());
          // if required, release the data position if the count is complete
          if (release_flag) {
            // increase the count
            std::get<1>(data_[ref.get_pos()])++;
            COUT << "zmq_data_service::server_func send data: release count increased: slot = " << ref.get_pos() << ", count=(" << std::get<1>(data_[ref.get_pos()]) << ", " << release_count << ")" << ENDL;
            if (std::get<1>(data_[ref.get_pos()]) >= release_count) {
              // count is complete -> release
              std::get<0>(data_[ref.get_pos()]) = false;
              std::get<1>(data_[ref.get_pos()]) = 0;
              std::get<2>(data_[ref.get_pos()]) = std::vector<unsigned char>{};
              emptyList_.push_back(ref.get_pos());
              COUT << "zmq_data_service::server_func send data: slot released: slot = " << ref.get_pos() << ENDL;
            }
          }
          COUT << "zmq_data_service::server_func send data: size = " << std::get<2>(data_[ref.get_pos()]).size() << " slot = " << ref.get_pos() << ENDL;
        } else {
          localServer_[next].send("", 0);
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
