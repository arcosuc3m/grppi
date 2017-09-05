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

#include <gtest/gtest.h>

#include "map.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class map_test : public ::testing::Test {
public:
  T execution_;
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
  void run_nary(const E & e) {
    grppi::map(e, 
      v.begin(), v.end(), w.begin(),
      [this](int x, int y, int z) { 
        invocations++; 
        return x+y+z; 
      },
      v2.begin(), v3.begin()
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
    EXPECT_EQ(1, this->invocations); // one invocation
    EXPECT_EQ(84, this->w[0]);
  }

  void setup_multiple_unary() {
    v = vector<int>{1,2,3,4,5};
    w = vector<int>(5);
    expected = vector<int>{2,4,6,8,10};
  }

  void check_multiple_unary() {
    EXPECT_EQ(5, this->invocations); // five invocations
    EXPECT_TRUE(equal(begin(this->expected), end(this->expected), begin(this->w)));
  }

  void setup_single_nary() {
    v = vector<int>{11};
    v2 = vector<int>{22};
    v3 = vector<int>{33};
    w = vector<int>{99};
  }

  void check_single_nary() {
    EXPECT_EQ(1, this->invocations); // one invocation
    EXPECT_EQ(66, this->w[0]);
  }

  void setup_multiple_nary() {
    v = vector<int>{1,2,3,4,5};
    v2 = vector<int>{2,4,6,8,10};
    v3 = vector<int>{10,10,10,10,10};
    w = vector<int>(5);
    expected = vector<int>{13,16,19,22,25};
  }

  void check_multiple_nary() {
    EXPECT_EQ(5, this->invocations); // five invocations
    EXPECT_TRUE(equal(begin(this->expected), end(this->expected), begin(this->w)));
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

TYPED_TEST(map_test, dyn_empty_unary)
{
  this->setup_empty();
  this->run_unary(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(map_test, static_single_unary)
{
  this->setup_single_unary();
  this->run_unary(this->execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, dyn_single_unary)
{
  this->setup_single_unary();
  this->run_unary(this->dyn_execution_);
  this->check_single_unary();
}

TYPED_TEST(map_test, static_multiple_unary)
{
  this->setup_multiple_unary();
  this->run_unary(this->execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, dyn_multiple_unary)
{
  this->setup_multiple_unary();
  this->run_unary(this->dyn_execution_);
  this->check_multiple_unary();
}

TYPED_TEST(map_test, static_empty_nary)
{
  this->setup_empty();
  this->run_nary(this->execution_);
  this->check_empty();
}

TYPED_TEST(map_test, dyn_empty_nary)
{
  this->setup_empty();
  this->run_nary(this->dyn_execution_);
  this->check_empty();
}


TYPED_TEST(map_test, static_single_nary)
{
  this->setup_single_nary();
  this->run_nary(this->execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, dyn_single_nary)
{
  this->setup_single_nary();
  this->run_nary(this->dyn_execution_);
  this->check_single_nary();
}

TYPED_TEST(map_test, static_multiple_nary)
{
  this->setup_multiple_nary();
  this->run_nary(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(map_test, dyn_multiple_nary)
{
  this->setup_multiple_nary();
  this->run_nary(this->execution_);
  this->check_multiple_nary();
}
