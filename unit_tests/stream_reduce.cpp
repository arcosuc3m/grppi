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

#include "grppi/pipeline.h"
#include "grppi/stream_reduce.h"
#include "grppi/dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class stream_reduce_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int out{};
  int window{};
  int offset{};

  // Vectors
  vector<int> v{};
  
  // Invocation counter
  std::atomic<int> invocations_gen{0};
  std::atomic<int> invocations_kernel{0};
  std::atomic<int> invocations_reduce{0};

  template <typename E>
  void run_reduction_add(const E & e) {
    grppi::pipeline(e,
      [this]() -> grppi::optional<int> {
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
TYPED_TEST_SUITE(stream_reduce_test, executions,);

// Check functionality with empty stream and sink function
TYPED_TEST(stream_reduce_test, static_empty) //NOLINT
{ 
  this->setup_empty();
  this->run_reduction_add(this->execution_);
  this->check_empty();
}

TYPED_TEST(stream_reduce_test, dyn_empty) //NOLINT
{ 
  this->setup_empty();
  this->run_reduction_add(this->dyn_execution_);
  this->check_empty();
}

// Process single element
TYPED_TEST(stream_reduce_test, static_single) //NOLINT
{ 
  this->setup_single();
  this->run_reduction_add(this->execution_);
  this->check_single();
}

TYPED_TEST(stream_reduce_test, dyn_single) //NOLINT
{ 
  this->setup_single();
  this->run_reduction_add(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(stream_reduce_test, static_multiple) //NOLINT
{ 
  this->setup_multiple();
  this->run_reduction_add(this->execution_);
  this->check_multiple();
}

TYPED_TEST(stream_reduce_test, dyn_multiple) //NOLINT
{ 
  this->setup_multiple();
  this->run_reduction_add(this->dyn_execution_);
  this->check_multiple();
}


// Process multiple elements with changes in the window and offset parameters
TYPED_TEST(stream_reduce_test, static_window_offset) //NOLINT
{ 
  this->setup_window_offset();
  this->run_reduction_add(this->execution_);
  this->check_window_offset();
}

TYPED_TEST(stream_reduce_test, dyn_window_offset) //NOLINT
{ 
  this->setup_window_offset();
  this->run_reduction_add(this->dyn_execution_);
  this->check_window_offset();
}

// Process multiple elements with changes in the window and offset parameters
TYPED_TEST(stream_reduce_test, static_offset_window) //NOLINT
{
  this->setup_offset_window();
  this->run_reduction_add(this->execution_);
  this->check_offset_window();
}

TYPED_TEST(stream_reduce_test, dyn_offset_window) //NOLINT
{
  this->setup_offset_window();
  this->run_reduction_add(this->dyn_execution_);
  this->check_offset_window();
}
