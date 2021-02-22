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

#include "grppi/reduce.h"
#include "grppi/dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class reduce_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int out{};

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
  
  template <typename E>
  void run_unary_range(const E & e) {
    out = grppi::reduce(e, v, 0,
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
TYPED_TEST_SUITE(reduce_test, executions,);

TYPED_TEST(reduce_test, static_single) //NOLINT
{
  this->setup_single();
  this->run_unary(this->execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, static_single_size) //NOLINT
{
  this->setup_single();
  this->run_unary_size(this->execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, static_single_range) //NOLINT
{
  this->setup_single();
  this->run_unary_range(this->execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, dyn_single) //NOLINT
{
  this->setup_single();
  this->run_unary(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, dyn_single_size) //NOLINT
{
  this->setup_single();
  this->run_unary_size(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, dyn_single_range) //NOLINT
{
  this->setup_single();
  this->run_unary_range(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(reduce_test, static_multiple) //NOLINT
{
  this->setup_multiple();
  this->run_unary_size(this->execution_);
  this->check_multiple();
}

TYPED_TEST(reduce_test, static_multiple_size) //NOLINT
{
  this->setup_multiple();
  this->run_unary_size(this->execution_);
  this->check_multiple();
}

TYPED_TEST(reduce_test, static_multiple_range) //NOLINT
{
  this->setup_multiple();
  this->run_unary_range(this->execution_);
  this->check_multiple();
}

TYPED_TEST(reduce_test, dyn_multiple) //NOLINT
{
  this->setup_multiple();
  this->run_unary_size(this->dyn_execution_);
  this->check_multiple();
}

TYPED_TEST(reduce_test, dyn_multiple_size) //NOLINT
{
  this->setup_multiple();
  this->run_unary_size(this->dyn_execution_);
  this->check_multiple();
}

TYPED_TEST(reduce_test, dyn_multiple_range) //NOLINT
{
  this->setup_multiple();
  this->run_unary_range(this->dyn_execution_);
  this->check_multiple();
}
