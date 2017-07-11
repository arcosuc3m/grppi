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
#include <atomic>
#include <utility>

#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <vector>
#include "common/parallel_execution_native.h"

using namespace std;
using namespace grppi;

TEST(numa_affinity, set_apply){
  parallel_execution_native p(2);
  p.set_numa_affinity(0,{0});
  p.set_numa_affinity(1,{1});
  std::vector<std::thread> task;
  std::vector<int> ids(2);
  for (auto i = 0;i<2;++i) {
    task.push_back(std::thread(
      [&](){
        p.register_thread();
        ids[p.get_thread_id()] = 10;
        p.deregister_thread();
      }
    ));
  }
  for(auto &t:task) t.join();
  EXPECT_EQ(10,ids[0]);
  EXPECT_EQ(10,ids[1]);
}


TEST(thread_affinity, set_apply){
 parallel_execution_native p(2);
  p.set_thread_affinity(0,{0});
  p.set_thread_affinity(1,{1});
  std::vector<std::thread> task;
  std::vector<int> ids(2);
  for (auto i = 0;i<2;++i) {
    task.push_back(std::thread(
      [&](){
        p.register_thread();
        ids[p.get_thread_id()] = 10;
        p.deregister_thread();
      }
    ));
  }
  for(auto &t:task) t.join();
  EXPECT_EQ(10,ids[0]);
  EXPECT_EQ(10,ids[1]);
}

TEST(numa_affinity, resize_threads){
 parallel_execution_native p(2);
  p.set_numa_affinity(0,{0});
  p.set_numa_affinity(1,{1});
  std::vector<std::thread> task;
  std::vector<int> ids(2);
  int n_thr = p.get_num_threads();
  for (auto i = 0;i<2;++i) {
    task.push_back(std::thread(
      [&](){
        p.register_thread();
        ids[p.get_thread_id()] = 10;
        p.deregister_thread();
      }
    ));
  }
  for(auto &t:task) t.join();
  p.set_num_threads(1);  
  int n_thrb = p.get_num_threads();
  
  EXPECT_EQ(2,n_thr);
  EXPECT_EQ(1,n_thrb);
  EXPECT_EQ(10,ids[0]);
  EXPECT_EQ(10,ids[1]);
}

TEST(thread_affinity, resize_threads){
  parallel_execution_native p(2);
  p.set_thread_affinity(0,{0});
  p.set_thread_affinity(1,{1});
  int n_thr = p.get_num_threads();
  std::vector<std::thread> task;
  std::vector<int> ids(2);
  for (auto i = 0;i<2;++i) {
    task.push_back(std::thread(
      [&](){
        p.register_thread();
        ids[p.get_thread_id()] = 10;
        p.deregister_thread();
      }
    ));
  }
  for(auto &t:task) t.join();
  p.set_num_threads(1);  
  int n_thrb = p.get_num_threads();
  
  EXPECT_EQ(2,n_thr);
  EXPECT_EQ(1,n_thrb);
  EXPECT_EQ(10,ids[0]);
  EXPECT_EQ(10,ids[1]);


}

