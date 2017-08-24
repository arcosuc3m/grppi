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
#include <experimental/optional>


#include <gtest/gtest.h>

#include "pipeline.h"
#include "farm.h"

#include "supported_executions.h"
#include "poly/polymorphic_execution.h"

#include <iostream>
#include <numeric>

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class pipeline_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Variables
  int out;
  int counter;

  // Vectors
  vector<int> v{};
  vector<vector<int> > v2{};

  // Invocation counter
  std::atomic<int> invocations_init{0};
  std::atomic<int> invocations_last{0};
  std::atomic<int> invocations_intermediate{0};

  void setup_two_stages_empty() {
    counter=0;
  }

  template <typename E>
  void run_two_stages(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {}; 
      },
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  void check_two_stages_empty() {
    EXPECT_EQ(1, invocations_init); 
    EXPECT_EQ(0, invocations_last); 
  }

  void setup_two_stages() {
    out = 0;
    counter = 2;
  }

  void check_two_stages() {
    EXPECT_EQ(3, invocations_init); 
    EXPECT_EQ(2, invocations_last); 
    EXPECT_EQ(3, this->out);
  }

  void setup_three_stages() {
    counter = 5;
    out = 0;
  }

  template <typename E>
  void run_three_stages(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {}; 
      },
      [this](int x) {
        invocations_intermediate++;
        return x*2;
      },
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  void check_three_stages() {
    EXPECT_EQ(6, invocations_init); 
    EXPECT_EQ(5, invocations_last); 
    EXPECT_EQ(5, invocations_intermediate);
    EXPECT_EQ(30, this->out);
  }

  void setup_composed() {
    counter = 5;
    out = 0;
  }

  template <typename E>
  void run_composed(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      grppi::pipeline(
        [this](int x) {
          invocations_intermediate++;
          return x*x;
        },
        [this](int x) {
          return x+1;
        }
      ),
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  template <typename E>
  void run_composed_piecewise(const E & e) {
    auto inner = grppi::pipeline(
      [this](int x) {
        invocations_intermediate++;
        return x*x;
      },
      [this](int x) {
        return x+1;
      });

    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      inner,
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  void check_composed() {
    EXPECT_EQ(6, invocations_init); 
    EXPECT_EQ(5, invocations_last); 
    EXPECT_EQ(5, invocations_intermediate);
    EXPECT_EQ(60, out);
  }

};

// Test for execution policies defined in supported_executions.h
//TYPED_TEST_CASE(pipeline_test, executions);
using executions_tmp = ::testing::Types<
  grppi::sequential_execution,
  grppi::parallel_execution_native>;

TYPED_TEST_CASE(pipeline_test, executions_tmp);

TYPED_TEST(pipeline_test, static_two_stages_empty)
{
  this->setup_two_stages_empty();
  this->run_two_stages(this->execution_);
  this->check_two_stages_empty();
}

TYPED_TEST(pipeline_test, poly_two_stages_empty)
{
  this->setup_two_stages_empty();
  this->run_two_stages(this->poly_execution_);
  this->check_two_stages_empty();
}

TYPED_TEST(pipeline_test, static_two_stages)
{
  this->setup_two_stages();
  this->run_two_stages(this->execution_);
  this->check_two_stages();
}

TYPED_TEST(pipeline_test, poly_two_stages)
{
  this->setup_two_stages();
  this->run_two_stages(this->poly_execution_);
  this->check_two_stages();
}

TYPED_TEST(pipeline_test, static_three_stages)
{
  this->setup_three_stages();
  this->run_three_stages(this->execution_);
  this->check_three_stages();
}

TYPED_TEST(pipeline_test, poly_three_stages)
{
  this->setup_three_stages();
  this->run_three_stages(this->poly_execution_);
  this->check_three_stages();
}

TYPED_TEST(pipeline_test, static_composed)
{
  this->setup_composed();
  this->run_composed(this->execution_);
  this->check_composed();
}

TYPED_TEST(pipeline_test, static_composed_piecewise)
{
  this->setup_composed();
  this->run_composed_piecewise(this->execution_);
  this->check_composed();
}
