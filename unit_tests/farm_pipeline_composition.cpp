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

#include "grppi/farm.h"
#include "grppi/pipeline.h"
#include "grppi/dyn/dynamic_execution.h"
#include "grppi/common/optional.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template<typename T>
class farm_pipeline_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Variables
  std::atomic<int> output{};

  // Vectors
  vector<int> v{};
  vector<int> v2{};
  vector<int> v3{};
  vector<int> w{};

  // entry counter
  size_t idx_in = 0;
  size_t idx_out = 0;

  // Invocation counter
  std::atomic<int> invocations_in{0};
  std::atomic<int> invocations_op{0};
  std::atomic<int> invocations_op2{0};
  std::atomic<int> invocations_sk{0};

  void setup_empty() {}

  template<typename E>
  void run_empty(const E & e)
  {
    grppi::pipeline(e,
        [this]() -> grppi::optional<int> {
          invocations_in++;
          return {};
        },
        grppi::farm(2,
            grppi::pipeline(
                [this](int x) {
                  invocations_op++;
                  return x;
                },
                [this](int x) {
                  invocations_op2++;
                  return x;
                }
            )
        ),
        [](int) {}
    );
  }

  void check_empty()
  {
    EXPECT_EQ(1, invocations_in); // Functor in was invoked once
    EXPECT_EQ(0, invocations_op); // Functor op was not invoked
    EXPECT_EQ(0, invocations_op2); // Functor op2 was not invoked
  }

  void setup_empty_sink() {}

  template<typename E>
  void run_empty_sink(const E & e)
  {
    grppi::pipeline(e,
        [this]() -> grppi::optional<int> {
          invocations_in++;
          return {};
        },
        grppi::farm(2,
            grppi::pipeline(
                [this](int x) {
                  invocations_op++;
                  return x;
                },
                [this](int x) {
                  invocations_op2++;
                  return x;
                }
            )
        ),
        [this](int) {
          this->invocations_sk++;
        }
    );
  }

  void check_empty_sink()
  {
    EXPECT_EQ(1, invocations_in);
    EXPECT_EQ(0, invocations_op);
    EXPECT_EQ(0, invocations_op2);
    EXPECT_EQ(0, invocations_sk);
  }

  void setup_empty_ary() {}

  template<typename E>
  void run_empty_ary(const E & e)
  {
    grppi::pipeline(e,
        [this]() -> grppi::optional<tuple<int, int, int>> {
          invocations_in++;
          return {};
        },
        grppi::farm(2,
            grppi::pipeline(
                [this](tuple<int, int, int> x) {
                  invocations_op++;
                  return x;
                },
                [this](tuple<int, int, int> x) {
                  invocations_op2++;
                  return x;
                })
        ),
        [&](tuple<int, int, int>) {
          invocations_sk++;
        }
    );
  }

  void check_empty_ary()
  {
    EXPECT_EQ(1, invocations_in);
    EXPECT_EQ(0, invocations_op);
    EXPECT_EQ(0, invocations_op2);
    EXPECT_EQ(0, invocations_sk);
  }

  void setup_single_sink()
  {
    v = vector<int>{42};
    w = vector<int>{99};
  }

  template<typename E>
  void run_single_sink(const E & e)
  {
    grppi::pipeline(e,
        [this]() -> grppi::optional<int> {
          invocations_in++;
          if (idx_in < v.size()) {
            idx_in++;
            return v[idx_in - 1];
          }
          else { return {}; }
        },
        grppi::farm(2,
            grppi::pipeline(
                [this](int x) {
                  invocations_op++;
                  return x * 2;
                },
                [this](int x) {
                  invocations_op2++;
                  return x * 2;
                }
            )
        ),
        [this](int x) {
          invocations_sk++;
          w[idx_out] = x;
          idx_out++;
        }
    );
  }

  void check_single_sink()
  {
    EXPECT_EQ(2, invocations_in); // Functor in was invoked once
    EXPECT_EQ(1, this->invocations_op); // one invocation of function op
    EXPECT_EQ(1, this->invocations_op2); // one invocation of function op
    EXPECT_EQ(1, this->invocations_sk); // one invocation of function sk
    EXPECT_EQ(168, this->w[0]);
  }

  void setup_single_ary_sink()
  {
    v = vector<int>{11};
    v2 = vector<int>{22};
    v3 = vector<int>{33};
    w = vector<int>{99};
  }

  template<typename E>
  void run_single_ary_sink(const E & e)
  {
    grppi::pipeline(e,
        [this]() -> grppi::optional<tuple<int, int, int>> {
          invocations_in++;
          if (idx_in < v.size()) {
            idx_in++;
            return make_tuple(v[idx_in - 1], v2[idx_in - 1], v3[idx_in - 1]);
          }
          else { return {}; }
        },
        grppi::farm(2,
            grppi::pipeline(
                [this](tuple<int, int, int> x) {
                  invocations_op++;
                  return get<0>(x) + get<1>(x) + get<2>(x);;
                },
                [this](int x) {
                  invocations_op2++;
                  return x * 2;
                }
            )
        ),
        [this](int x) {
          invocations_sk++;
          w[idx_out] = x;
          idx_out++;
        });
  }

  void check_single_ary_sink()
  {
    EXPECT_EQ(2, invocations_in); // Functor in was invoked twice
    EXPECT_EQ(1, this->invocations_op); // one invocation of function op
    EXPECT_EQ(1, this->invocations_op2); // one invocation of function op
    EXPECT_EQ(1, this->invocations_sk); // one invocation of function sk
    EXPECT_EQ(132, this->w[0]);
  }

  void setup_multiple_sink()
  {
    v = vector<int>{1, 2, 3, 4, 5};
    output = 0;
  }

  template<typename E>
  void run_multiple_sink(const E & e)
  {
    grppi::pipeline(e,
        [this]() -> grppi::optional<int> {
          invocations_in++;
          if (idx_in < v.size()) {
            idx_in++;
            return v[idx_in - 1];
          }
          else { return {}; }
        },
        grppi::farm(2,
            grppi::pipeline(
                [this](int x) {
                  invocations_op++;
                  return x * 2;
                },
                [this](int x) {
                  invocations_op2++;
                  return x * 2;
                }
            )
        ),
        [this](int x) {
          invocations_sk++;
          output += x;
        }
    );
  }

  void check_multiple_sink()
  {
    EXPECT_EQ(6, this->invocations_in); // six invocations of function in
    EXPECT_EQ(5, this->invocations_op); // five invocations of function op
    EXPECT_EQ(5, this->invocations_op2); // five invocations of function sk
    EXPECT_EQ(5, this->invocations_sk); // five invocations of function sk
    EXPECT_EQ(60, this->output);
  }

  void setup_multiple_ary_sink()
  {
    v = vector<int>{1, 2, 3, 4, 5};
    v2 = vector<int>{2, 4, 6, 8, 10};
    v3 = vector<int>{10, 10, 10, 10, 10};
    output = 0;
  }

  template<typename E>
  void run_multiple_ary_sink(const E & e)
  {
    grppi::pipeline(e,
        [this]() -> grppi::optional<tuple<int, int, int>> {
          invocations_in++;
          if (idx_in < v.size()) {
            idx_in++;
            return make_tuple(v[idx_in - 1], v2[idx_in - 1], v3[idx_in - 1]);
          }
          else {
            return {};
          }
        },
        grppi::farm(2,
            grppi::pipeline(
                [this](tuple<int, int, int> x) {
                  invocations_op++;
                  return get<0>(x) + get<1>(x) + get<2>(x);
                },
                [this](int x) {
                  invocations_op2++;
                  return x * 2;
                }
            )
        ),
        [this](int x) {
          invocations_sk++;
          output += x;
        });
  }

  void check_multiple_ary_sink()
  {
    EXPECT_EQ(6, this->invocations_in); // six invocations of function in
    EXPECT_EQ(5, this->invocations_op); // five invocations of function op
    EXPECT_EQ(5, this->invocations_op2); // five invocations of function op
    EXPECT_EQ(5, this->invocations_sk); // five invocations of function sk
    EXPECT_EQ(190, this->output);
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_SUITE(farm_pipeline_test, executions,); //NOLINT

TYPED_TEST(farm_pipeline_test, static_empty) //NOLINT
{
  this->setup_empty();
  this->run_empty(this->execution_);
  this->check_empty();
}

TYPED_TEST(farm_pipeline_test, dyn_empty) //NOLINT
{
  this->setup_empty();
  this->run_empty(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(farm_pipeline_test, static_empty_sink) //NOLINT
{
  this->setup_empty_sink();
  this->run_empty_sink(this->execution_);
  this->check_empty_sink();
}

TYPED_TEST(farm_pipeline_test, dyn_empty_sink) //NOLINT
{
  this->setup_empty_sink();
  this->run_empty_sink(this->dyn_execution_);
  this->check_empty_sink();
}

TYPED_TEST(farm_pipeline_test, static_empty_ary) //NOLINT
{
  this->setup_empty();
  this->run_empty_ary(this->execution_);
  this->check_empty();
}

TYPED_TEST(farm_pipeline_test, dyn_empty_ary) //NOLINT
{
  this->setup_empty();
  this->run_empty_ary(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(farm_pipeline_test, static_single_sink) //NOLINT
{
  this->setup_single_sink();
  this->run_single_sink(this->execution_);
  this->check_single_sink();
}

TYPED_TEST(farm_pipeline_test, dyn_single_sink) //NOLINT
{
  this->setup_single_sink();
  this->run_single_sink(this->dyn_execution_);
  this->check_single_sink();
}

TYPED_TEST(farm_pipeline_test, static_single_ary_sink) //NOLINT
{
  this->setup_single_ary_sink();
  this->run_single_ary_sink(this->dyn_execution_);
  this->check_single_ary_sink();
}

TYPED_TEST(farm_pipeline_test, dyn_single_ary_sink) //NOLINT
{
  this->setup_single_ary_sink();
  this->run_single_ary_sink(this->dyn_execution_);
  this->check_single_ary_sink();
}

TYPED_TEST(farm_pipeline_test, static_multiple_sink) //NOLINT
{
  this->setup_multiple_sink();
  this->run_multiple_sink(this->execution_);
  this->check_multiple_sink();
}

TYPED_TEST(farm_pipeline_test, dyn_multiple_sink) //NOLINT
{
  this->setup_multiple_sink();
  this->run_multiple_sink(this->dyn_execution_);
  this->check_multiple_sink();
}

TYPED_TEST(farm_pipeline_test, static_multiple_ary_sink) //NOLINT
{
  this->setup_multiple_ary_sink();
  this->run_multiple_ary_sink(this->execution_);
  this->check_multiple_ary_sink();
}

TYPED_TEST(farm_pipeline_test, dyn_multiple_ary_sink) //NOLINT
{
  this->setup_multiple_ary_sink();
  this->run_multiple_ary_sink(this->dyn_execution_);
  this->check_multiple_ary_sink();
}