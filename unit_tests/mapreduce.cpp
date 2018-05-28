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
#include <iostream>

#include "mapreduce.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class map_reduce_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int output{};

  // Vectors
  vector<int> v{};
  vector<int> v2{};

  // Invocation counter
  std::atomic<int> invocations_transformer{0};

  template <typename E>
  auto run_square_sum(const E & e) {
    return grppi::map_reduce(e, v.begin(), v.end(), 0,
      [this](int x) { 
        invocations_transformer++; 
        return x*x;
      },
      [](int x, int y) { 
        return x + y; 
      }
    );
  }

  template <typename E>
  auto run_square_sum_size(const E & e) {
    return grppi::map_reduce(e, begin(v), v.size(), 0,
      [this](int x) { 
        invocations_transformer++; 
        return x*x;
      },
      [](int x, int y) { 
        return x + y; 
      }
    );
  }

  template <typename E>
  auto run_square_sum_range(const E & e) {
    return grppi::map_reduce(e, v, 0,
      [this](int x) { 
        invocations_transformer++; 
        return x*x;
      },
      [](int x, int y) { 
        return x + y; 
      }
    );
  }

  template <typename E>
  auto run_scalar_product_tuple_iter(const E & e){
    return  grppi::map_reduce(e,
      make_tuple(v.begin(),v2.begin()), v.end(), 0,
      [this](int x1, int x2) {
        invocations_transformer++;
        return x1 * x2;
      },
      [](int x, int y){
        return x + y;
      }
    );
  }

  template <typename E>
  auto run_scalar_product_tuple_size(const E & e){
    return  grppi::map_reduce(e,
      make_tuple(v.begin(),v2.begin()), v.size(), 0,
      [this](int x1, int x2){
        invocations_transformer++;
        return x1 * x2;
      },
      [](int x, int y){
        return x + y;
      }
    );
  }

  template <typename E>
  auto run_scalar_product_tuple_range(const E & e){
    return  grppi::map_reduce(e,
      grppi::zip(v,v2), 0,
      [this](int x1, int x2) {
        invocations_transformer++;
        return x1 * x2;
      },
      [](int x, int y){
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
    EXPECT_EQ(1, this->output);
  }

  void setup_multiple() {
    v = vector<int>{1,2,3,4,5};
    output = 0;
  }

  void check_multiple() {
    EXPECT_EQ(5, this->invocations_transformer);
    EXPECT_EQ(55, this->output);
  }

  void setup_single_scalar_product() {
    v = vector<int>{5};
    v2 = vector<int>{6};
    output = 0;
  }

  void check_single_scalar_product(){
    EXPECT_EQ(1, this->invocations_transformer);
    EXPECT_EQ(30, this->output);
  }

  void setup_multiple_scalar_product() {
    v = vector<int>{1,2,3,4,5};
    v2 = vector<int>{2,4,6,8,10};
    output = 0;
  }

  void check_multiple_scalar_product() {
    EXPECT_EQ(5, this->invocations_transformer);
    EXPECT_EQ(110, this->output);
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(map_reduce_test, executions);

TYPED_TEST(map_reduce_test, static_single_square_sum)
{
  this->setup_single();
  this->output = this->run_square_sum(this->execution_);
  this->check_single();
}

TYPED_TEST(map_reduce_test, static_single_square_sum_size)
{
  this->setup_single();
  this->output = this->run_square_sum_size(this->execution_);
  this->check_single();
}

TYPED_TEST(map_reduce_test, static_single_square_sum_range)
{
  this->setup_single();
  this->output = this->run_square_sum_range(this->execution_);
  this->check_single();
}

TYPED_TEST(map_reduce_test, dyn_single_square_sum)
{
  this->setup_single();
  this->output = this->run_square_sum(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(map_reduce_test, dyn_single_square_sum_size)
{
  this->setup_single();
  this->output = this->run_square_sum_size(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(map_reduce_test, dyn_single_square_sum_range)
{
  this->setup_single();
  this->output = this->run_square_sum_range(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(map_reduce_test, static_multiple_square_sum)
{
  this->setup_multiple();
  this->output = this->run_square_sum(this->execution_);
  this->check_multiple();
}

TYPED_TEST(map_reduce_test, static_multiple_square_sum_size)
{
  this->setup_multiple();
  this->output = this->run_square_sum_size(this->execution_);
  this->check_multiple();
}

TYPED_TEST(map_reduce_test, static_multiple_square_sum_range)
{
  this->setup_multiple();
  this->output = this->run_square_sum_range(this->execution_);
  this->check_multiple();
}

TYPED_TEST(map_reduce_test, dyn_multiple_square_sum)
{
  this->setup_multiple();
  this->output = this->run_square_sum(this->dyn_execution_);
  this->check_multiple();
}

TYPED_TEST(map_reduce_test, dyn_multiple_square_sum_size)
{
  this->setup_multiple();
  this->output = this->run_square_sum_size(this->dyn_execution_);
  this->check_multiple();
}

TYPED_TEST(map_reduce_test, dyn_multiple_square_sum_range)
{
  this->setup_multiple();
  this->output = this->run_square_sum_range(this->dyn_execution_);
  this->check_multiple();
}

TYPED_TEST(map_reduce_test, static_single_scalar_product_tuple_iter)
{
  this->setup_single_scalar_product();
  this->output = this->run_scalar_product_tuple_iter(this->execution_);
  this->check_single_scalar_product();
}

TYPED_TEST(map_reduce_test, static_single_scalar_product_tuple_size)
{
  this->setup_single_scalar_product();
  this->output = this->run_scalar_product_tuple_size(this->execution_);
  this->check_single_scalar_product();
}

TYPED_TEST(map_reduce_test, static_single_scalar_product_tuple_range)
{
  this->setup_single_scalar_product();
  this->output = this->run_scalar_product_tuple_range(this->execution_);
  this->check_single_scalar_product();
}

TYPED_TEST(map_reduce_test, dyn_single_scalar_product_tuple_iter)
{
  this->setup_single_scalar_product();
  this->output = this->run_scalar_product_tuple_iter(this->dyn_execution_);
  this->check_single_scalar_product();
}

TYPED_TEST(map_reduce_test, dyn_single_scalar_product_tuple_size)
{
  this->setup_single_scalar_product();
  this->output = this->run_scalar_product_tuple_size(this->dyn_execution_);
  this->check_single_scalar_product();
}

TYPED_TEST(map_reduce_test, dyn_single_scalar_product_tuple_range)
{
  this->setup_single_scalar_product();
  this->output = this->run_scalar_product_tuple_range(this->dyn_execution_);
  this->check_single_scalar_product();
}

TYPED_TEST(map_reduce_test, static_multiple_scalar_product_tuple_iter)
{
  this->setup_multiple_scalar_product();
  this->output = this->run_scalar_product_tuple_iter(this->execution_);
  this->check_multiple_scalar_product();
}

TYPED_TEST(map_reduce_test, static_multiple_scalar_product_tuple_size)
{
  this->setup_multiple_scalar_product();
  this->output = this->run_scalar_product_tuple_size(this->execution_);
  this->check_multiple_scalar_product();
}

TYPED_TEST(map_reduce_test, static_multiple_scalar_product_tuple_range)
{
  this->setup_multiple_scalar_product();
  this->output = this->run_scalar_product_tuple_range(this->execution_);
  this->check_multiple_scalar_product();
}

TYPED_TEST(map_reduce_test, dyn_multiple_scalar_product_tuple_iter)
{
  this->setup_multiple_scalar_product();
  this->output = this->run_scalar_product_tuple_iter(this->dyn_execution_);
  this->check_multiple_scalar_product();
}

TYPED_TEST(map_reduce_test, dyn_multiple_scalar_product_tuple_size)
{
  this->setup_multiple_scalar_product();
  this->output = this->run_scalar_product_tuple_size(this->dyn_execution_);
  this->check_multiple_scalar_product();
}

TYPED_TEST(map_reduce_test, dyn_multiple_scalar_product_tuple_size_range)
{
  this->setup_multiple_scalar_product();
  this->output = this->run_scalar_product_tuple_range(this->dyn_execution_);
  this->check_multiple_scalar_product();
}
