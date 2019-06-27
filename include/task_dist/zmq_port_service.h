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
#include <stdexcept>
#include <utility>

#include <zmq.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

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
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & machine_id_;
        ar & key_;
        ar & wait_;
    }
    int machine_id_;
    int key_;
    bool wait_;
public:
    zmq_port_key() {}
    zmq_port_key(int machine_id, int key, bool wait) :
        machine_id_(machine_id), key_(key), wait_(wait)
    {}
    int get_id() {return machine_id_;}
    int get_key() {return key_;}
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

  zmq_port_service(std::string server, int port, bool is_server) :
    server_(server),
    port_(port),
    is_server_(is_server),
    context_(1),
    requestSock_(context_,ZMQ_REQ)
  {
    //std::cout << "zmq_port_service 1" << std::endl;

    // connect request socket
    std::ostringstream ss;
    ss << tcpConnectPattern[0] << server_ << tcpConnectPattern[1] << port_;
    requestSock_.connect(ss.str());

    // if server, bind reply socket and launch thread
    if (is_server_) {
      // server thread launched
      server_thread_ = std::thread(&zmq_port_service::server_func, this);
    }
    //std::cout << "zmq_port_service end" << std::endl;
  }

  /**
  \brief Port Service destructor. Also finish the data server if exists.
  */
  ~zmq_port_service()
  {
    //std::cout << "~zmq_port_service begin" << std::endl;
    if (is_server_) {
        ////std::cout << "join proxy_thread_" << std::endl;
        requestSock_.send(endCmd.data(), endCmd.size(), 0);
        server_thread_.join();
    }
    //std::cout << "~zmq_port_service end" << std::endl;
  }

  /**
  \brief Get the port number from the server and key
  \param machine_id_ id of the server machine.
  \param key key for this port.
  \param wait wait until the port is set or not.
  \return port number desired.
  */
  int get (int machine_id_, int key, bool wait) throw(std::runtime_error)
  {
    // send the command tag
    requestSock_.send(getCmd.data(), getCmd.size(), ZMQ_SNDMORE);
    //std::cout << "get send cmd GET" << std::endl;

    
    // serialize obj into an std::string
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
    boost::archive::binary_oarchive oa(os);

    zmq_port_key portkey(machine_id_,key,wait);
    oa << portkey;
    os.flush();
    
    // send the reference (server_id,pos)
    requestSock_.send(serial_str.data(), serial_str.length());
    //std::cout << "get send portkey: (" << portkey.get_id() << "," << portkey.get_key() << ")" << std::endl;

    // receive the data
    zmq::message_t message;
    requestSock_.recv(&message);
    //std::cout << "get rec data: size=" << message.size() << std::endl;

    if (message.size() == 0) {
        //std::cout << "Error Item not found" << std::endl;
        throw std::runtime_error("Item not found");
    }
    
    // wrap buffer inside a stream and deserialize serial_str into obj
    boost::iostreams::basic_array_source<char> device((char *)message.data(), message.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > is(device);
    boost::archive::binary_iarchive ia(is);

    int item;
    try {
      ia >> item;
    } catch (...) {
        //std::cout << "Error Incorrect Type" << std::endl;
        throw std::runtime_error("Incorrect Type");
    }

    return item;
  }

  /**
  \brief Set the port number for this server and key
  \param machine_id_ id of the server machine.
  \param key key for this port.
  \param port number to store.
  */
  void set(int machine_id_, int key, int port) throw (std::runtime_error)
  {
    //std::cout << "set send id:" << server_id_ << std::endl;
    // send the command tag
    requestSock_.send(setCmd.data(), setCmd.size(), ZMQ_SNDMORE);
    //std::cout << "set send cmd SET" << std::endl;

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
        //std::cout << "Error Type not serializable" << std::endl;
        throw std::runtime_error("Type not serializable");
    }

    // send the data
    requestSock_.send(serial_str.data(), serial_str.length());
    //std::cout << "set send data: size=" << serial_str.length() << std::endl;

    // receive the reference (server_id,pos)
    zmq::message_t message;
    requestSock_.recv(&message);

   
    if (message.size() != 0) {
        //std::cout << "Error full data storage" << std::endl;
        throw std::runtime_error("Full Data Storage");
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
  int port_;
  bool is_server_;
  std::map<zmq_port_key, int> port_data_;
  zmq::context_t context_;
  zmq::socket_t requestSock_;
  std::map<std::pair<int,int>,int> portStorage_;
  std::map<std::pair<int,int>,std::vector<std::string>> waitQueue_;
  /// Proxy server address
  std::thread server_thread_;


  /**
  \brief Server function to store and release data form the storage array.
  */
  void server_func ()
  {
    
    //std::cout << "server_func begin" << std::endl;
    zmq::socket_t replySock_ = zmq::socket_t(context_,ZMQ_ROUTER);
    std::ostringstream ss;
    ss << tcpBindPattern[0] << port_ << tcpBindPattern[1];
    replySock_.bind(ss.str());

    while (1) {
      
      zmq::message_t msg;

      // receive client id
      replySock_.recv(&msg);
      std::string client_id((char *)msg.data(), msg.size());

      // recv zero frame
      replySock_.recv(&msg);
      
      // recv command
      replySock_.recv(&msg);

      
      // set command
      if ( (msg.size() == setCmd.size()) &&
           (0 == std::memcmp(msg.data(),static_cast<const void*>(setCmd.data()),setCmd.size())) ) {
        //std::cout << "server_func SET" << std::endl;
        
        // recv item and copy it to the map storage
        replySock_.recv(&msg);

        boost::iostreams::basic_array_source<char> device((char *)msg.data(), msg.size());
        boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
        boost::archive::binary_iarchive ia(s);
  
        zmq_port_key ref;
        int port;
        ia >> ref;
        ia >> port;
        
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
        //std::cout << "server_func GET" << std::endl;
        
        // recv item and copy it to the map storage
        replySock_.recv(&msg);

        int port = -1;
        try {
          boost::iostreams::basic_array_source<char> device((char *)msg.data(), msg.size());
          boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(device);
          boost::archive::binary_iarchive ia(s);
  
          zmq_port_key ref;
          ia >> ref;

          //std::cout << "server_func GET portkey: (" << ref.get_id() << "," << ref.get_key() << "," << ref.get_wait() <<")" << std::endl;
          try {
            port = portStorage_.at(std::make_pair(ref.get_id(),ref.get_key()));
          } catch (std::out_of_range &e) { // port is not stored
            if (ref.get_wait()) {
              //std::cout << "server_func GET WAIT" << std::endl;
              waitQueue_[std::make_pair(ref.get_id(),ref.get_key())].emplace_back(client_id);
            } else {
              //std::cout << "server_func GET NO WAIT" << std::endl;
              //std::cout << "server_func ERROR get: port not found" << std::endl;
              replySock_.send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
              replySock_.send("", 0, ZMQ_SNDMORE);
              replySock_.send("", 0);
            }
            continue; // process next message
          }
          //std::cout << "server_func GET port: " << port << std::endl;

    
          std::string serial_str;
          boost::iostreams::back_insert_device<std::string> inserter(serial_str);
          boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > os(inserter);
          boost::archive::binary_oarchive oa(os);

          oa << port;
          os.flush();

          //std::cout << "server_func GET port string: " << serial_str << std::endl;

          replySock_.send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
          replySock_.send("", 0, ZMQ_SNDMORE);
          replySock_.send(serial_str.data(), serial_str.length());
        } catch (...) {
          //std::cout << "server_func ERROR get" << std::endl;
          replySock_.send(client_id.data(), client_id.size(), ZMQ_SNDMORE);
          replySock_.send("", 0, ZMQ_SNDMORE);
          replySock_.send("", 0);
        }
      } else if ( (msg.size() == endCmd.size()) &&
        (0 == std::memcmp(msg.data(), static_cast<const void*>(endCmd.data()), endCmd.size())) ) {
        //std::cout << "server_func END" << std::endl;
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
    //std::cout << "server_func end" << std::endl;
  }
};

/**
@}
*/


}

#endif
