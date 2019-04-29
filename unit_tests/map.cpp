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

#include <gtest/gtest.h>

#include "grppi/map.h"
#include "grppi/dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class map_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Vectors
  vector<int> v{};
  vector<int> v2{};
  vector<int> v3{};
  vector<int> w{};
  vector<int> expected{};

  // Invocation counter
  std::atomic<int> invocations{0};

  template <typename E>
  void run_unary(const E & e) {
    grppi::map(e, 
      v.begin(), v.end(), w.begin(),
      [this](int i) {
        invocations++; 
        return i*2; 
      }
    );
  }

  template <typename E>
  void run_unary_size(const E & e) {
    grppi::map(e, 
      v.begin(), v.size(), w.begin(),
      [this](int i) {
        invocations++; 
        return i*2; 
      }
    );
  }

  template <typename E>
  void run_unary_range(const E & e) {
    grppi::map(e,
      v, w,
      [this](int i) {
        invocations++;
        return i*2;
      }
    );
  }

  template <typename E>
  void run_nary_tuple_iter(const E &e) {
    grppi::map(e, 
      make_tuple(v.begin(),v2.begin(),v3.begin()),
      v.end(), w.begin(),
      [this](int x, int y, int z) {
        invocations++; 
        return x+y+z; 
      }
    );
  }

  template<typename E>
  void run_nary_tuple_size(const E & e) {
    grppi::map(e,
      make_tuple(v.begin(),v2.begin(),v3.begin()),
      v.size(), w.begin(),
       [this](int x, int y, int z){
         invocations++;
         return x+y+z;
       }
    );
  }

  template<typename E>
  void run_nary_tuple_range(const E & e) {
    grppi::map(e,
      grppi::zip(v,v2,v3),
      w,
       [this](int x, int y, int z){
         invocations++;
         return x+y+z;
       }
    );
  }

  void setup_empty() {
  }

  void check_empty() {
    ASSERT_EQ(0, invocations); // Functor was not invoked
  }

  void setup_single_unary() {
    v = vector<int>{42};
    w = vector<int>{99};
  }

  void check_single_unary() {
    EXPECT_EQ(1, invocations); // one invocation
    EXPECT_EQ(84, w[0]);
  }

  void setup_multiple_unary() {
    v = vector<int>{1,2,3,4,5};
    w = vector<int>(5);
    expected = vector<int>{2,4,6,8,10};
  }

  void check_multiple_unary() {
    EXPECT_EQ(5, invocations); // five invocations
    EXPECT_TRUE(equal(begin(expected), end(expected), begin(w)));
  }

  void setup_single_nary() {
    v = vector<int>{11};
    v2 = vector<int>{22};
    v3 = vector<int>{33};
    w = vector<int>{99};
  }

  void check_single_nary() {
    EXPECT_EQ(1, invocations); // one invocation
    EXPECT_EQ(66, w[0]);
  }

  void setup_multiple_nary() {
    v = vector<int>{1,2,3,4,5};
    v2 = vector<int>{2,4,6,8,10};
    v3 = vector<int>{10,10,10,10,10};
    w = vector<int>(5);
    expected = vector<int>{13,16,19,22,25};
  }

  void check_multiple_nary() {
    EXPECT_EQ(5, invocations); // five invocations
    EXPECT_TRUE(equal(begin(expected), end(expected), begin(w)));
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(map_test, executions);

TYPED_TEST(map_test, static_empty_unary)
{
  this->setup_empty();
  this->run_unary(this->execution_);
  this->check_empty();
}

TYPED_TEST(map_test, static_empty_unary_size)
{
  this->setup_empty();
  this->run_unary_size(this->execution_);
  this->check_empty();
}

TYPED_TEST(map_test, static_empty_unary_range)
{
  this->setup_empty();
  this->run_unary_range(this->execution_);
  this->check_empty();
}

TYPED_TEST(map_test, dyn_empty_unary)
{
  this->setup_empty();
  this->run_unary(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(map_test, dyn_empty_unary_size)
{
  this->setup_empty();
  this->run_unary_size(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(map_test, dyn_empty_unary_range)
{
  this->setup_empty();
  this->run_unary_range(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(map_test, static_single_unary)
{
  this->setup_single_unary();
  this->run_unary(this->execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, static_single_unary_size)
{
  this->setup_single_unary();
  this->run_unary_size(this->execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, static_single_unary_range)
{
  this->setup_single_unary();
  this->run_unary_range(this->execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, dyn_single_unary)
{
  this->setup_single_unary();
  this->run_unary(this->dyn_execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, dyn_single_unary_size)
{
  this->setup_single_unary();
  this->run_unary_size(this->dyn_execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, dyn_single_unary_range)
{
  this->setup_single_unary();
  this->run_unary_range(this->dyn_execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, static_multiple_unary)
{
  this->setup_multiple_unary();
  this->run_unary(this->execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, static_multiple_unary_size)
{
  this->setup_multiple_unary();
  this->run_unary_size(this->execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, static_multiple_unary_range)
{
  this->setup_multiple_unary();
  this->run_unary_range(this->execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, dyn_multiple_unary)
{
  this->setup_multiple_unary();
  this->run_unary(this->dyn_execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, dyn_multiple_unary_size)
{
  this->setup_multiple_unary();
  this->run_unary_size(this->dyn_execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, dyn_multiple_unary_range)
{
  this->setup_multiple_unary();
  this->run_unary_range(this->dyn_execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, static_empty_nary_tuple_iter)
{
  this->setup_empty();
  this->run_nary_tuple_iter(this->execution_);
  this->check_empty();
}

TYPED_TEST(map_test, static_empty_nary_tuple_size)
{
  this->setup_empty();
  this->run_nary_tuple_size(this->execution_);
  this->check_empty();
}

TYPED_TEST(map_test, static_empty_nary_tuple_range)
{
  this->setup_empty();
  this->run_nary_tuple_range(this->execution_);
  this->check_empty();
}

TYPED_TEST(map_test, dyn_empty_nary_tuple_iter)
{
  this->setup_empty();
  this->run_nary_tuple_iter(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(map_test, dyn_empty_nary_tuple_size)
{
  this->setup_empty();
  this->run_nary_tuple_size(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(map_test, dyn_empty_nary_tuple_range)
{
  this->setup_empty();
  this->run_nary_tuple_range(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(map_test, static_single_nary_tuple_iter)
{
  this->setup_single_nary();
  this->run_nary_tuple_iter(this->execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, static_single_nary_tuple_size)
{
  this->setup_single_nary();
  this->run_nary_tuple_size(this->execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, static_single_nary_tuple_range)
{
  this->setup_single_nary();
  this->run_nary_tuple_range(this->execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, dyn_single_nary_tuple_iter)
{
  this->setup_single_nary();
  this->run_nary_tuple_iter(this->dyn_execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, dyn_single_nary_tuple_size)
{
  this->setup_single_nary();
  this->run_nary_tuple_size(this->dyn_execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, dyn_single_nary_tuple_range)
{
  this->setup_single_nary();
  this->run_nary_tuple_range(this->dyn_execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, static_multiple_nary_tuple_iter)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_iter(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(map_test, static_multiple_nary_tuple_size)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_size(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(map_test, static_multiple_nary_tuple_range)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_range(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(map_test, dyn_multiple_nary_tuple_iter)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_iter(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(map_test, dyn_multiple_nary_tuple_size)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_size(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(map_test, dyn_multiple_nary_tuple_range)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_range(this->execution_);
  this->check_multiple_nary();
}

