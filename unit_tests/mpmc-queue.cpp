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
#include <gtest/gtest.h>
#include "grppi/common/mpmc_queue.h"

#include <thread>
#include <atomic>
#include <utility>
#include <iostream>
#include <numeric>

using namespace std;
using namespace grppi;

template <typename T>
class mpmc_test : public ::testing::Test {
public:
  using queue_type = T;

  auto make_queue(int sz) { return queue_type{sz}; }
  void push_value(queue_type & q) { q.push(1); }
  auto pop_value(queue_type & q) { return q.pop(); }

};

using types = ::testing::Types<atomic_mpmc_queue<int>, locked_mpmc_queue<int>>;

TYPED_TEST_SUITE(mpmc_test, types);
                
TYPED_TEST(mpmc_test, constructor) { //NOLINT
  auto q = this->make_queue(10);
  EXPECT_TRUE(q.empty());
}

TYPED_TEST(mpmc_test, push_pop){ //NOLINT
  auto q = this->make_queue(10);
  this->push_value(q);
  EXPECT_FALSE(q.empty());

  auto value = this->pop_value(q);
  EXPECT_TRUE(q.empty());
  EXPECT_EQ(1,value);
}

TYPED_TEST(mpmc_test, concurrent_push_pop){ //NOLINT
  auto q = this->make_queue(3);

  std::vector<std::thread> thrs;
  for (int i=0; i<6; ++i) {
    thrs.push_back(std::thread([&q,i](){ 
      q.push(i);
    }));
  }

  int val = 0;
  for (int i=0; i<6; ++i) {
    val += q.pop();
  }

  for (auto & t : thrs) { t.join(); }

  EXPECT_EQ(15, val);
  EXPECT_TRUE(q.empty());
}

TYPED_TEST(mpmc_test , concurrent_pop_push){ //NOLINT
  auto q = this->make_queue(3);

  std::vector<std::thread> thrs;
  std::vector<int> v(6);
  for (int i=0; i<6; ++i) {
    thrs.push_back(std::thread([&q,&v,i](){
      v[i] = q.pop();
    }));
  }

  for (int i=0; i<6; ++i) {
    q.push(i);
  }

  for (auto & t : thrs) {
    t.join();
  }

  int val = std::accumulate(begin(v), end(v),0);

  EXPECT_EQ(15, val);
  EXPECT_TRUE(q.empty());
}


TEST(mpmc_queue_blocking, constructor){
  mpmc_queue<int> queue(10, queue_mode::blocking);
  EXPECT_TRUE(queue.empty());
}


TEST(mpmc_queue_lockfree, constructor){
  mpmc_queue<int> queue(10, queue_mode::lockfree);
  EXPECT_TRUE(queue.empty());
}

TEST(mpmc_queue_blocking, push_pop){
  mpmc_queue<int> queue(10, queue_mode::blocking);
  queue.push(1);
  EXPECT_FALSE(queue.empty());

  int value = queue.pop();
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(1,value);
}

TEST(mpmc_queue_lockfree, push_pop){
  mpmc_queue<int> queue(10, queue_mode::lockfree);
  queue.push(1);
  EXPECT_FALSE(queue.empty());

  int value = queue.pop();
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(1,value);
}

TEST(mpmc_queue_blocking, concurrent_push_pop){
  mpmc_queue<int> q(3, queue_mode::blocking);
  std::vector<std::thread> thrs;
  for (int i = 0; i<6; ++i) {
    thrs.push_back(std::thread([&q,i](){
      q.push(i);
    }));
  } 

  int val = 0;
  for (int i=0; i<6; ++i) {
    val += q.pop();
  }

  for (auto & t : thrs) {
    t.join();
  }

  EXPECT_EQ(15, val);
  EXPECT_TRUE(q.empty());
}


TEST(mpmc_queue_lockfree, concurrent_push_pop){
  mpmc_queue<int> q(3, queue_mode::lockfree);
  std::vector<std::thread> thrs;
  for (auto i=0; i<6; ++i) {
    thrs.push_back(std::thread([&,i](){
      q.push(i);
    }));
  } 

  int val = 0;
  for (int i=0; i<6; ++i) {
    val += q.pop();
  }

  for (auto & t : thrs) {
    t.join();
  }
  EXPECT_EQ(15, val);
  EXPECT_TRUE(q.empty());
}

TEST(mpmc_queue_blocking, concurrent_pop_push){
  mpmc_queue<int> q(3, queue_mode::blocking);
  std::vector<std::thread> thrs;
  std::vector<int> v(6);
  for (int i=0; i<6; ++i) {
    thrs.push_back(std::thread([&q,&v,i](){
      v[i] = q.pop();
    }));
  }

  for (int i=0; i<6; ++i) {
    q.push(i);
  }

  for (auto & t : thrs) {
    t.join();
  }
  int val = std::accumulate(std::begin(v), std::end(v), 0);

  EXPECT_EQ(15, val);
  EXPECT_TRUE(q.empty());
}



TEST(mpmc_queue_lockfree, concurrent_pop_push){
  mpmc_queue<int> q(3, queue_mode::lockfree);
  std::vector<std::thread> thrs;
  std::vector<int> v(6);

  for (int i=0; i<6; ++i) {
    thrs.push_back(std::thread([&v,&q,i](){
      v[i] = q.pop();
    }));
  }
  for (auto i = 0;i<6; ++i) {
    q.push(i);
  }

  for (auto & t : thrs) {
    t.join();
  }
  auto val = std::accumulate(std::begin(v), std::end(v), 0);

  EXPECT_EQ(15, val);
  EXPECT_TRUE(q.empty());
}

