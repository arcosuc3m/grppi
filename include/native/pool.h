/**
* @version		GrPPI v0.2
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef GRPPI_NATIVE_POOL_H
#define GRPPI_NATIVE_POOL_H

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

namespace grppi {

class thread_pool
{
  private: 
    boost::thread_group threadpool;
    boost::asio::io_service ioService;
    std::vector<boost::asio::io_service::work> works;//(ioService);
    int busy_threads= 0;

  public:
    thread_pool (){
    };
    
    template <typename T> 
     void create_task(T task ) { ioService.post(task); }

    void initialise (int num_threads) {
       works.push_back( std::move(boost::asio::io_service::work(ioService)));
       for(unsigned int nthr= 0; nthr < num_threads; nthr++){
         threadpool.create_thread(
            boost::bind(&boost::asio::io_service::run, &ioService)
         );
       }
    };
  
    ~thread_pool(){
        ioService.stop();
        threadpool.join_all();
     };

};

}

#endif
