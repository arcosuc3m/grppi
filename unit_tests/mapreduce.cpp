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
#include <iostream>

#include "mapreduce.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class map_reduce_test : public ::testing::Test {
public:
  T execution_;
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int output;

  // Vectors
  vector<int> v{};

  // Invocation counter
  std::atomic<int> invocations_transformer{0};

  template <typename E>
  auto run_square_sum(const E & e) {
    return grppi::map_reduce(e, v.begin(), v.end(), 0,
      [this](int x) { 
        invocations_transformer++; 
        return x*2; 
      },
      [](int x, int y) { 
        return x + y; 
      }
    );
  }

  void setup_single() {
    v = vector<int>{1};
    output = 0;
  }

  void check_single() {
    EXPECT_EQ(1, invocations_transformer); 
    EXPECT_EQ(2, this->output);
  }

  void setup_multiple() {
    v = vector<int>{1,2,3,4,5};
    output = 0;
  }

  void check_multiple() {
    EXPECT_EQ(5, this->invocations_transformer);
    EXPECT_EQ(30, this->output);
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(map_reduce_test, executions);

TYPED_TEST(map_reduce_test, static_single)
{
  this->setup_single();
  this->output = this->run_square_sum(this->execution_);
  this->check_single();
}

TYPED_TEST(map_reduce_test, dyn_single)
{
  this->setup_single();
  this->output = this->run_square_sum(this->dyn_execution_);
  this->check_single();
}



TYPED_TEST(map_reduce_test, static_multiple)
{
  this->setup_multiple();
  this->output = this->run_square_sum(this->execution_);
  this->check_multiple();
}

TYPED_TEST(map_reduce_test, dyn_multiple)
{
  this->setup_multiple();
  this->output = this->run_square_sum(this->dyn_execution_);
  this->check_multiple();
}
