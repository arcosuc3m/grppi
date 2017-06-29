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

#include "reduce.h"
#include "common/polymorphic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class reduce_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Variables
  int out;

  // Vectors
  vector<int> v{};

  void setup_empty() {
    out = 0;
  }

  void check_empty() {
    EXPECT_EQ(0, this->out);
  }

  void setup_single() {
    out = 0;
    v = vector<int>{1};
  }

  void check_single() {
    EXPECT_EQ(1, this->out);
  }

  void setup_multiple() {
    out = 0;
    v = vector<int>{1,2,3,4,5};
  }

  void check_multiple() {
    EXPECT_EQ(15, this->out); 
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(reduce_test, executions);

TYPED_TEST(reduce_test, static_empty)
{
  this->setup_empty();
  this->out = grppi::reduce(this->execution_, begin(this->v), end(this->v), std::plus<int>());
  this->check_empty();
}

TYPED_TEST(reduce_test, static_single)
{
  this->setup_single();
  this->out = grppi::reduce(this->execution_, begin(this->v), end(this->v), std::plus<int>());
  this->check_single();
}

TYPED_TEST(reduce_test, static_multiple)
{
  this->setup_multiple();
  this->out = grppi::reduce(this->execution_, begin(this->v), end(this->v), std::plus<int>());
  this->check_multiple();
}
