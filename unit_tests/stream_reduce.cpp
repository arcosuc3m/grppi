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
#include <experimental/optional>

#include <gtest/gtest.h>

#include "pipeline.h"
#include "stream_reduce.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class stream_reduce_test : public ::testing::Test {
public:
  T execution_;
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int out;
  int window;
  int offset;

  // Vectors
  vector<int> v{};
  
  // Invocation counter
  std::atomic<int> invocations_gen{0};
  std::atomic<int> invocations_kernel{0};
  std::atomic<int> invocations_reduce{0};

  template <typename E>
  void run_reduction_add(const E & e) {
    grppi::pipeline(e,
      [this]() -> optional<int> { 
        invocations_gen++; 
        if(v.size() > 0){
          auto problem = v.back();
          v.pop_back();
          return problem;
      }
      else return {};
    },
    grppi::reduce(window, offset, 0,
      [](int x, int y) { return x+y; }),
    [this](int x) { 
      std::cerr << "consuming " << x << "\n";
      invocations_reduce++;
      out += x;
    });
  }

  void setup_empty() {
    window = 3;
    offset = 3;
  }

  void check_empty() {
    EXPECT_EQ(1, invocations_gen);
    EXPECT_EQ(0, invocations_kernel);
    EXPECT_EQ(0, invocations_reduce);
  }

  void setup_single() {
    out = 0;
    v = vector<int>{1};
    window = 1;
    offset = 0;
  }

  void check_single() {
    EXPECT_EQ(2, invocations_gen);
    EXPECT_EQ(1, invocations_reduce);
    EXPECT_EQ(1, this->out);
  }

  void setup_multiple() {
    out = 0;
    v = vector<int>{1,2,3,4,5};
    window = 1;
    offset = 1;
  }

  void check_multiple() {
    EXPECT_EQ(6, invocations_gen);
    EXPECT_EQ(5, invocations_reduce);
    EXPECT_EQ(15, this->out);
  }

  void setup_window_offset() {
    out = 0;
    v = vector<int>{1,2,3,4,5,6};
    window = 2;
    offset = 1;
  }

  void check_window_offset() {
    EXPECT_EQ(7, invocations_gen);
    EXPECT_EQ(5, invocations_reduce);
    EXPECT_EQ(35, this->out);
  }


  void setup_offset_window() {
    out = 0;
    v = vector<int>{1,2,3,4,5,6,7,8,9,10};
    window = 2;
    offset = 4;
  }

  void check_offset_window() {
    EXPECT_EQ(11, invocations_gen);
    EXPECT_EQ(3, invocations_reduce);
    EXPECT_EQ(33, this->out);
  }
};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(stream_reduce_test, executions);

// Check functionality with empty stream and sink function
TYPED_TEST(stream_reduce_test, static_empty)
{ 
  this->setup_empty();
  this->run_reduction_add(this->execution_);
  this->check_empty();
}

TYPED_TEST(stream_reduce_test, dyn_empty)
{ 
  this->setup_empty();
  this->run_reduction_add(this->dyn_execution_);
  this->check_empty();
}

// Process single element
TYPED_TEST(stream_reduce_test, static_single)
{ 
  this->setup_single();
  this->run_reduction_add(this->execution_);
  this->check_single();
}

TYPED_TEST(stream_reduce_test, dyn_single)
{ 
  this->setup_single();
  this->run_reduction_add(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(stream_reduce_test, static_multiple)
{ 
  this->setup_multiple();
  this->run_reduction_add(this->execution_);
  this->check_multiple();
}

TYPED_TEST(stream_reduce_test, dyn_multiple)
{ 
  this->setup_multiple();
  this->run_reduction_add(this->dyn_execution_);
  this->check_multiple();
}


// Process multiple elements with changes in the window and offset parameters
TYPED_TEST(stream_reduce_test, static_window_offset)
{ 
  this->setup_window_offset();
  this->run_reduction_add(this->execution_);
  this->check_window_offset();
}

TYPED_TEST(stream_reduce_test, dyn_window_offset)
{ 
  this->setup_window_offset();
  this->run_reduction_add(this->dyn_execution_);
  this->check_window_offset();
}

// Process multiple elements with changes in the window and offset parameters
TYPED_TEST(stream_reduce_test, static_offset_window)
{
  this->setup_offset_window();
  this->run_reduction_add(this->execution_);
  this->check_offset_window();
}

TYPED_TEST(stream_reduce_test, dyn_offset_window)
{
  this->setup_offset_window();
  this->run_reduction_add(this->dyn_execution_);
  this->check_offset_window();
}
