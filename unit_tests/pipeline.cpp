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
#if __cplusplus < 201703L
#include <experimental/optional>
#else
#include <optional>
#endif
#include <numeric>

#include <gtest/gtest.h>

#include "pipeline.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

#if __cplusplus < 201703L
template <typename T>
using optional = std::experimental::optional<T>;
#endif

template <typename T>
class pipeline_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int out{};
  int counter{};

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
    EXPECT_EQ(30, out);
  }

  void setup_composed_last() {
    counter = 5;
    out = 0;
  }

  template <typename E>
  void run_composed_last(const E & e) {
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
          invocations_last++;
          out +=x;
        }
      ));
  }

  template <typename E>
  void run_composed_last_piecewise(const E & e) {
    auto inner = grppi::pipeline(
        [this](int x) {
          invocations_intermediate++;
          return x*x;
        },
        [this](int x) {
          invocations_last++;
          out +=x;
        }
    );
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      inner);
  }


  void check_composed_last() {
    EXPECT_EQ(6, invocations_init); 
    EXPECT_EQ(5, invocations_last); 
    EXPECT_EQ(5, invocations_intermediate);
    EXPECT_EQ(55, out);
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
        [](int x) {
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
      [](int x) {
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
TYPED_TEST_CASE(pipeline_test, executions);

TYPED_TEST(pipeline_test, static_two_stages_empty)
{
  this->setup_two_stages_empty();
  this->run_two_stages(this->execution_);
  this->check_two_stages_empty();
}

TYPED_TEST(pipeline_test, dyn_two_stages_empty)
{
  this->setup_two_stages_empty();
  this->run_two_stages(this->dyn_execution_);
  this->check_two_stages_empty();
}

TYPED_TEST(pipeline_test, static_two_stages)
{
  this->setup_two_stages();
  this->run_two_stages(this->execution_);
  this->check_two_stages();
}

TYPED_TEST(pipeline_test, dyn_two_stages)
{
  this->setup_two_stages();
  this->run_two_stages(this->dyn_execution_);
  this->check_two_stages();
}

TYPED_TEST(pipeline_test, static_three_stages)
{
  this->setup_three_stages();
  this->run_three_stages(this->execution_);
  this->check_three_stages();
}

TYPED_TEST(pipeline_test, dyn_three_stages)
{
  this->setup_three_stages();
  this->run_three_stages(this->dyn_execution_);
  this->check_three_stages();
}

TYPED_TEST(pipeline_test, static_composed_last)
{
  this->setup_composed_last();
  this->run_composed_last(this->execution_);
  this->check_composed_last();
}

TYPED_TEST(pipeline_test, static_composed_piecewise_last)
{
  this->setup_composed_last();
  this->run_composed_last_piecewise(this->execution_);
  this->check_composed_last();
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
