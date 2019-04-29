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

#include "grppi/stream_pool.h"
#include "grppi/dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class stream_pool_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Variables
  std::vector<std::vector<int>> population;
  int out{};

  // Invocation counter
  std::atomic<int> invocations_sel{0};
  std::atomic<int> invocations_evol{0};
  std::atomic<int> invocations_eval{0};
  std::atomic<int> invocations_end{0};

  void setup_problem() {
    for( int i = 0; i < 5 ; i++ ){
      population.push_back({i,5});
    }
    out = 0; 
  }

  void check_problem() {
    EXPECT_LE(5, invocations_sel);
    EXPECT_LE(5, invocations_evol);
    EXPECT_LE(5, invocations_eval);
    EXPECT_LE(5, invocations_end);
    EXPECT_EQ(9, out);
  }

  void setup_empty() {
    out = 0;
  }

  void check_empty() {
    EXPECT_EQ(0, invocations_sel);
    EXPECT_EQ(0, invocations_evol);
    EXPECT_EQ(0, invocations_eval);
    EXPECT_EQ(0, invocations_end);
    EXPECT_EQ(0, out);
  }
  
  template <typename E>
  void run_stream_pool(const E & e){
    grppi::stream_pool(e, population,
      [this] ( auto & population ) {
        invocations_sel++;
        auto individual = population[0];
        population.erase(population.begin());
        return individual; 
      },
      [this](auto individual){
        invocations_evol++;
        return individual;
      },
      [this](auto evolved, auto ){
        invocations_eval++;
        return evolved;
      },
      [this](auto evolved){
        invocations_end++;
        auto cost = evolved[0] + evolved[1];
        if( cost > out ) out = cost;
        return out == 9;
      }
    );
  }

};

TYPED_TEST_CASE(stream_pool_test, executions);

TYPED_TEST(stream_pool_test, static_empty)
{
  this->setup_empty();
  this->run_stream_pool(this->execution_);
  this->check_empty();
}

TYPED_TEST(stream_pool_test, dyn_empty)
{
  this->setup_empty();
  this->run_stream_pool(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(stream_pool_test, static_problem)
{
  this->setup_problem();
  this->run_stream_pool(this->execution_);
  this->check_problem();
}

TYPED_TEST(stream_pool_test, dyn_problem)
{
  this->setup_problem();
  this->run_stream_pool(this->dyn_execution_);
  this->check_problem();
}
