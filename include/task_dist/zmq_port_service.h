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

#ifndef GRPPI_ZMQ_PORT_SERVICE_H
#define GRPPI_ZMQ_PORT_SERVICE_H

#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <atomic>
#include <stdexcept>
#include <utility>
#include <memory>
#include <sstream>
#include <exception>

#include <zmq.hpp>

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
//#pragma GCC diagnostic pop

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

/**
\defgroup zmq_port_service ZeroMQ zmq port service
\brief Port service support types.
@{
*/


class zmq_port_key
{
private:
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned long version)
    {
        if (version >= 0) {
          ar & machine_id_;
          ar & key_;
          ar & wait_;
        }
    }
    long machine_id_;
    long key_;
    bool wait_;
public:
    zmq_port_key() {}
    zmq_port_key(long machine_id, long key, bool wait) :
        machine_id_(machine_id), key_(key), wait_(wait)
    {}
    long get_id() {return machine_id_;}
    long get_key() {return key_;}
    bool get_wait() {return wait_;}
};

class zmq_port_service {
public:
   
  // no copy constructors
  zmq_port_service(const zmq_port_service&) =delete;
  zmq_port_service& operator=(zmq_port_service const&) =delete;

  /**
  \brief Port Service contructor interface. Also creates one port server if req.
  \param server port server name.
  \param port port for the port sever.
  \param is_server request to create a port server
  */

  zmq_port_service(std::string server, long port, bool is_server) :
    server_(server),
    port_(port),
    is_server_(is_server),
    context_(1)
  {
    COUT << "zmq_port_service 1" << ENDL;

    // if server, bind reply socket and launch thread
    if (is_server_) {
      // server thread launched
      server_thread_ = std::thread(&zmq_port_service::server_func, this);
    }
    COUT << "zmq_port_service end" << ENDL;
  }

  /**
  \brief Port Service destructor. Also finish the data server if exists.
  */
  ~zmq_port_service()
  {
    COUT << "~zmq_port_service begin" << ENDL;
    if (is_server_) {
        COUT << "join proxy_thread_" << ENDL;
        // Get the socket for this thread
        while(accessSockMap.test_and_set());
        if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
            requestSockList_.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(std::this_thread::get_id()),
                                    std::forward_as_tuple(create_socket()));
        }
        std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
        accessSockMap.clear();
        requestSock_->send(endCmd.data(), endCmd.size(), 0);
        server_thread_.join();
    }
    COUT << "~zmq_port_service end" << ENDL;
  }

  /**
  \brief return a new port to be used
  \return port number desired.
  */
  long new_port ()
  {
    return actual_port_number_++;
  }
  
  /**
  \brief Get the port number from the server and key
  \param machine_id_ id of the server machine.
  \param key key for this port.
  \param wait wait until the port is set or not.
  \return port number desired.
  */
  long get (long machine_id_, long key, bool wait)
  {
  
    // Get the socket for this thread
    while(accessSockMap.test_and_set());
    if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
        requestSockList_.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(std::this_thread::get_id()),
                                 std::forward_as_tuple(create_socket()));
    }
    std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
    accessSockMap.clear();

    // send the command tag
    COUT << "zmq_port_service::get (machine_id_,key,wait): (" << machine_id_ << ", " << key << ", " << wait << ")" << ENDL;
    requestSock_->send(getCmd.data(), getCmd.size(), ZMQ_SNDMORE);
    COUT << "zmq_port_service::get send cmd GET" << ENDL;

    
    // serialize obj into an std::string
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
    boost::archive::binary_oarchive oa(os);

    zmq_port_key portkey(machine_id_,key,wait);
    oa << portkey;
    os.flush();
    
    // send the reference (server_id,pos)
    requestSock_->send(serial_str.data(), serial_str.length());
    COUT << "zmq_port_service::get send portkey: (" << portkey.get_id() << "," << portkey.get_key() << ")" << ENDL;

    // receive the data
    zmq::message_t message;
    requestSock_->recv(&message);
    COUT << "zmq_port_service::get rec data: size=" << message.size() << ENDL;

    if (message.size() == 0) {
        COUT << "zmq_port_service::get Error Item not found" << ENDL;
        throw std::runtime_error("zmq_port_service::get: Item not found");
    }
    
    // wrap buffer inside a stream and deserialize serial_str into obj
    boost::iostreams::basic_array_source<char> device((char *)message.data(), message.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > is(device);
    boost::archive::binary_iarchive ia(is);

    long item;
    try {
      ia >> item;
    } catch (...) {
        COUT << "zmq_port_service::get Error Incorrect Type" << ENDL;
        std::throw_with_nested( std::runtime_error("zmq_port_service::get: Incorrect Type"));
    }

    return item;
  }

  /**
  \brief Set the port number for this server and key
  \param machine_id_ id of the server machine.
  \param key key for this port.
  \param port number to store.
  */
  void set(long machine_id_, long key, long port)
  {
    // Get the socket for this thread
    while(accessSockMap.test_and_set());
    if (requestSockList_.find(std::this_thread::get_id()) == requestSockList_.end()) {
        requestSockList_.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(std::this_thread::get_id()),
                                 std::forward_as_tuple(create_socket()));
    }
    std::shared_ptr<zmq::socket_t> requestSock_= requestSockList_.at(std::this_thread::get_id());
    accessSockMap.clear();

    COUT << "zmq_port_service::set (machine_id_,key,port): (" << machine_id_ << ", " << key << ", " << port << ")" << ENDL;
    // send the command tag
    requestSock_->send(setCmd.data(), setCmd.size(), ZMQ_SNDMORE);
    COUT << "zmq_port_service::set send cmd SET" << ENDL;

    // serialize obj into an std::string
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
    boost::archive::binary_oarchive oa(os);

    zmq_port_key portkey(machine_id_,key,false);
    try {
        oa << portkey;
        oa << port;
        os.flush();
    } catch (...) {
        COUT << "zmq_port_service::set Error Type not serializable" << ENDL;
        std::throw_with_nested( std::runtime_error("zmq_port_service::set  Type not serializable"));
    }

    // send the data
    COUT << "zmq_port_service::set send begin" << ENDL;
    requestSock_->send(serial_str.data(), serial_str.length());
    COUT << "zmq_port_service::set send data: size=" << serial_str.length() << ENDL;

    // receive the reference (server_id,pos)
    zmq::message_t message;
    requestSock_->recv(&message);

   
    if (message.size() != 0) {
        COUT << "zmq_port_service::set Error full data storage" << ENDL;
        throw std::runtime_error("zmq_port_service::set: Full Data Storage");
    }
    return;
  }

private:
  /// tcp bind pattern
  const std::vector<std::string> tcpBindPattern {"tcp://*:", ""};
  /// tcp connect pattern
  const std::vector<std::string> tcpConnectPattern {"tcp://", ":"};


  /// tag for set command
  const std::string setCmd{"SET"};
  /// tag for get command
  const std::string getCmd{"GET"};
  /// tag for end command
  const std::string endCmd{"END"};


  std::string server_;
  long port_;
  bool is_server_;
  std::map<zmq_port_key, long> port_data_;
  zmq::context_t context_;
  std::map<std::thread::id, std::shared_ptr<zmq::socket_t>> requestSockList_;
  std::map<std::pair<long,long>,long> portStorage_;
  std::map<std::pair<long,long>,std::vector<std::string>> waitQueue_;
  /// Proxy server address
  std::thread server_thread_;
  //mutual exclusion data for socket map structure
  std::atomic_flag accessSockMap = ATOMIC_FLAG_INIT;


  /// actual port number to be delivered
  long actual_port_number_{0};

  /**
  \brief Function to create a zmq request socket for the port service
  \return Shared pointer with the zmq socket.
  */
  std::shared_ptr<zmq::socket_t> create_socket ()
  {
    COUT << "zmq_port_service::create_socket begin" << ENDL;
    
    // create rquest socket shared pointer
    std::shared_ptr<zmq::socket_t> requestSock_ = std::make_shared<zmq::socket_t>(context_,ZMQ_REQ);

    // connect request socket
    std::ostringstream ss;
    ss << tcpConnectPattern[0] << server_ << tcpConnectPattern[1] << port_;
    COUT << "zmq_port_service::create_socket connect: " << ss.str() << ENDL;
    requestSock_->connect(ss.str());

    return requestSock_;
  }

  /**
  \brief Server function to store and release data form the storage array.
  */
  void server_func ()
  {
    
    COUT << "zmq_port_service::server_func begin" << ENDL;
    zmq::socket_t replySock_ = zmq::socket_t(context_,ZMQ_ROUTER);
    std::ostringstream ss;
    ss << tcpBindPattern[0] << port_ << tcpBindPattern[1];
    COUT << "zmq_port_service::server_func bind: " << ss.str() << ENDL;
    replySock_.bind(ss.str());

    while (1) {
      
      zmq::message_t msg;

      COUT << "zmq_port_service::server_func: replySock_.recv begin" << ENDL;

      // receive client id
      try {
        replySock_.recv(&msg);
      } catch (...) {
        std::cerr << "zmq_port_service::server_func: ERROR : replySock_.recv" << std::endl;
      }
      std::string client_id((char *)msg.data(), msg.size());
      COUT << "zmq_port_service::server_func: replySock_.recv client_id: " << client_id << ENDL;

      // recv zero frame
      replySock_.recv(&msg);
      
      // recv command
      replySock_.recv(&msg);

      COUT << "zmq_port_service::server_func: replySock_.recv cmd received" << ENDL;

      // set command
      if ( (msg.size() == setCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(setCmd.data()),setCmd.size())) ) {
        COUT << "zmq_port_service::server_func SET" << ENDL;

        // recv item and copy it to the map storage
        replySock_.recv(&msg);

        COUT << "zmq_port_service::server_func SET received" << ENDL;
        
        boost::iostreams::basic_array_source<char> device((char *)msg.data(), msg.size());
        boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
        boost::archive::binary_iarchive ia(s);
  
        zmq_port_key ref;
        long port;
        ia >> ref;
        ia >> port;
        
        COUT << "zmq_port_service::server_func SET portkey: (" << ref.get_id() << "," << ref.get_key() << "," << ref.get_wait() <<")" << ENDL;

        // insert or subsitute port in portkey
        portStorage_[std::make_pair(ref.get_id(),ref.get_key())] = port;
        
        // set ack
        replySock_.send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
        replySock_.send("", 0, ZMQ_SNDMORE);
        replySock_.send("", 0);
        
        //check if other client are waiting for this port
        auto wait_list = waitQueue_[std::make_pair(ref.get_id(),ref.get_key())];
        for (auto it = wait_list.begin(); it != wait_list.end(); it++) {
        
          replySock_.send(it->data(), it->size(), ZMQ_SNDMORE);
          replySock_.send("", 0, ZMQ_SNDMORE);

          std::string serial_str;
          boost::iostreams::back_insert_device<std::string> inserter(serial_str);
          boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
          boost::archive::binary_oarchive oa(os);

          oa << port;
          os.flush();

          replySock_.send(serial_str.data(), serial_str.length());
        }
      } else if ( (msg.size() == getCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(getCmd.data()), getCmd.size())) ) {
        COUT << "zmq_port_service::server_func GET" << ENDL;
        
        // recv item and copy it to the map storage
        replySock_.recv(&msg);

        long port = -1;
        try {
          boost::iostreams::basic_array_source<char> device((char *)msg.data(), msg.size());
          boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
          boost::archive::binary_iarchive ia(s);
  
          zmq_port_key ref;
          ia >> ref;

          COUT << "zmq_port_service::server_func GET portkey: (" << ref.get_id() << "," << ref.get_key() << "," << ref.get_wait() <<")" << ENDL;
          try {
            port = portStorage_.at(std::make_pair(ref.get_id(),ref.get_key()));
          } catch (std::out_of_range &e) { // port is not stored
            if (ref.get_wait()) {
              COUT << "zmq_port_service::server_func GET WAIT" << ENDL;
              waitQueue_[std::make_pair(ref.get_id(),ref.get_key())].emplace_back(client_id);
            } else {
              COUT << "zmq_port_service::server_func GET NO WAIT" << ENDL;
              COUT << "zmq_port_service::server_func ERROR get: port not found" << ENDL;
              replySock_.send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
              replySock_.send("", 0, ZMQ_SNDMORE);
              replySock_.send("", 0);
            }
            continue; // process next message
          }
          COUT << "zmq_port_service::server_func GET port: " << port << ENDL;

    
          std::string serial_str;
          boost::iostreams::back_insert_device<std::string> inserter(serial_str);
          boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
          boost::archive::binary_oarchive oa(os);

          oa << port;
          os.flush();

          COUT << "zmq_port_service::server_func GET port string: " << serial_str << ENDL;

          replySock_.send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
          replySock_.send("", 0, ZMQ_SNDMORE);
          replySock_.send(serial_str.data(), serial_str.length());
        } catch (...) {
          COUT << "zmq_port_service::server_func ERROR get" << ENDL;
          replySock_.send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
          replySock_.send("", 0, ZMQ_SNDMORE);
          replySock_.send("", 0);
        }
      } else if ( (msg.size() == endCmd.size()) &&
        (0 == std::memcmp(msg.data(), static_cast<const void*>(endCmd.data()), endCmd.size())) ) {
        COUT << "zmq_port_service::server_func END" << ENDL;
        // answer all pending requests with zero messsage
        for (auto it1 = waitQueue_.begin(); it1 != waitQueue_.end(); it1++) {
          for (auto it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
            replySock_.send(it2->data(), it2->size(), ZMQ_SNDMORE);
            replySock_.send("", 0, ZMQ_SNDMORE);
            replySock_.send("", 0);
          }
        }
        break;
      }
    }
    // need to release sockets???
    COUT << "zmq_port_service::server_func end" << ENDL;
  }
};

/**
@}
*/


}

#endif
