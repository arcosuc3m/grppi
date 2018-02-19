/**
* @version		GrPPI v0.3
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

#include "reduce.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class reduce_test : public ::testing::Test {
public:
  T execution_;
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int out;

  // Vectors
  vector<int> v{};

  template <typename E>
  void run_unary(const E & e) {
    out = grppi::reduce(e, v.begin(), v.end(), 0,
     [](int x, int y){ return x + y; }
    );
  }

  template <typename E>
  void run_unary_size(const E & e) {
    out = grppi::reduce(e, v.begin(), v.size(), 0,
     [](int x, int y){ return x + y; }
    );
  }
  
  void setup_single() {
    out = 0;
    v = vector<int>{1};
  }

  void check_single() {
    EXPECT_EQ(1, out);
  }

  void setup_multiple() {
    out = 0;
    v = vector<int>{1,2,3,4,5};
  }

  void check_multiple() {
    EXPECT_EQ(15, out); 
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(reduce_test, executions);

TYPED_TEST(reduce_test, static_single)
{
  this->setup_single();
  this->run_unary(this->execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, dyn_single)
{
  this->setup_single();
  this->run_unary(this->dyn_execution_);
  this->check_single();
}



TYPED_TEST(reduce_test, static_multiple)
{
  this->setup_multiple();
  this->run_unary_size(this->execution_);
  this->check_multiple();
}

TYPED_TEST(reduce_test, dyn_multiple)
{
  this->setup_multiple();
  this->run_unary_size(this->dyn_execution_);
  this->check_multiple();
}



TYPED_TEST(reduce_test, static_single_size)
{
  this->setup_single();
  this->run_unary_size(this->execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, dyn_single_size)
{
  this->setup_single();
  this->run_unary_size(this->dyn_execution_);
  this->check_single();
}



TYPED_TEST(reduce_test, static_multiple_size)
{
  this->setup_multiple();
  this->run_unary_size(this->execution_);
  this->check_multiple();
}

TYPED_TEST(reduce_test, dyn_multiple_size)
{
  this->setup_multiple();
  this->run_unary_size(this->dyn_execution_);
  this->check_multiple();
}