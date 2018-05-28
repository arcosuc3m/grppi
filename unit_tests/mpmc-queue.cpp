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
#include <atomic>
#include <utility>

#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include "common/mpmc_queue.h"

using namespace std;
using namespace grppi;

TEST(mpmc_queue_blocking, constructor){
  mpmc_queue<int> queue(10, queue_mode::blocking);
  EXPECT_EQ(true,queue.is_empty());
}


TEST(mpmc_queue_lockfree, constructor){
  mpmc_queue<int> queue(10, queue_mode::lockfree);
  EXPECT_EQ(true,queue.is_empty());
}

TEST(mpmc_queue_blocking, push_pop){
  mpmc_queue<int> queue(10, queue_mode::blocking);
  auto inserted = queue.push(1);
  EXPECT_EQ(true, inserted);
  EXPECT_EQ(false, queue.is_empty());
  auto value = queue.pop();
  EXPECT_EQ(true, queue.is_empty());
  EXPECT_EQ(1,value);
}

TEST(mpmc_queue_lockfree, push_pop){
  mpmc_queue<int> queue(10, queue_mode::lockfree);
  auto inserted = queue.push(1);
  EXPECT_EQ(true, inserted);
  EXPECT_EQ(false, queue.is_empty());
  auto value = queue.pop();
  EXPECT_EQ(true, queue.is_empty());
  EXPECT_EQ(1,value);
}

TEST(mpmc_queue_blocking, concurrent_push_pop){
mpmc_queue<int> queue(3, queue_mode::blocking);
 std::vector<std::thread> thrs;
 for(auto i = 0; i<6; i++){
   thrs.push_back(std::thread([&,i](){queue.push(i);} ));
 }
 auto val = 0;
 for(auto i = 0;i<6; i++){
   val += queue.pop();
 }
 for(auto & t : thrs) t.join();
 EXPECT_EQ(15, val);
 EXPECT_EQ(true, queue.is_empty());
}


TEST(mpmc_queue_lockfree, concurrent_push_pop){
mpmc_queue<int> queue(3, queue_mode::lockfree);
 std::vector<std::thread> thrs;
 for(auto i = 0; i<6; i++){
   thrs.push_back(std::thread([&,i](){queue.push(i);} ));
 } 
 auto val = 0;
 for(auto i = 0;i<6; i++){
   val += queue.pop();
 }
 for(auto & t : thrs) t.join();
 EXPECT_EQ(15, val);
 EXPECT_EQ(true, queue.is_empty());
}

TEST(mpmc_queue_blocking, concurrent_pop_push){
mpmc_queue<int> queue(3, queue_mode::blocking);
 std::vector<std::thread> thrs;
 std::vector<int> values(6);

 for(auto i = 0; i<6; i++){
   thrs.push_back(std::thread([&,i](){values[i] = queue.pop();} ));
 }
 for(auto i = 0;i<6; i++){
   queue.push(i);
 }

 for(auto & t : thrs) t.join();
 auto val = 0;
 for( auto &v : values) val+=v;

 EXPECT_EQ(15, val);
 EXPECT_EQ(true, queue.is_empty());
}



TEST(mpmc_queue_lockfree, concurrent_pop_push){
mpmc_queue<int> queue(3, queue_mode::lockfree);
 std::vector<std::thread> thrs;
 std::vector<int> values(6);

 for(auto i = 0; i<6; i++){
   thrs.push_back(std::thread([&,i](){values[i] = queue.pop();} ));
 }
 for(auto i = 0;i<6; i++){
   queue.push(i);
 }

 for(auto & t : thrs) t.join();
 auto val = 0;
 for( auto &v : values) val+=v;

 EXPECT_EQ(15, val);
 EXPECT_EQ(true, queue.is_empty());
}



