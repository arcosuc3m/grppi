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
#include <experimental/optional>

#include <gtest/gtest.h>

#include "grppi/stream_filter.h"
#include "grppi/pipeline.h"
#include "grppi/dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class stream_filter_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Vectors
  vector<int> v{};
  vector<int> w{};
  vector<int> expected_even{};  
  vector<int> expected_odd{};

  // Entry counter
  size_t idx_in = 0;
  size_t idx_out = 0;
 
  // Invocation counter
  std::atomic<int> invocations_in{0};
  std::atomic<int> invocations_op{0};
  std::atomic<int> invocations_out{0};

  void setup_empty() {
  }

  template <typename E>
  void run_keep_empty(const E & e) {
    grppi::pipeline(e,
      [this]() -> optional<int> { 
        invocations_in++;
        return {}; 
    },
    grppi::keep(
      [this](int) {
        invocations_op++; 
        return true; 
      }),
    [this](int) {
      this->invocations_out++; 
    });
  }

  template <typename E>
  void run_discard_empty(const E & e) {
    grppi::pipeline(e,
      [this]() -> optional<int> { 
        invocations_in++;
        return {}; 
    },
    grppi::discard(
      [this](int) {
        invocations_op++; 
        return true; 
      }),
    [this](int) {
      invocations_out++; 
    });
  }

  void check_keep_empty() {
    ASSERT_EQ(1, invocations_in); // Functor in was invoked only once
    ASSERT_EQ(0, invocations_op); // Functor op was not invoked
    ASSERT_EQ(0, invocations_out); // Functor out was not invoked
  }

  void check_discard_empty() {
    ASSERT_EQ(1, invocations_in); // Functor in was invoked only once
    ASSERT_EQ(0, invocations_op); // Functor op was not invoked
    ASSERT_EQ(0, invocations_out); // Functor out was not invoked
  }

  void setup_single() {
    v = vector<int>{42};
    w = vector<int>{99};
  }

  template <typename E>
  void run_keep_single(const E & e) {
    grppi::pipeline (e,
    [this]() -> optional<int> {
      invocations_in++;
      if(idx_in < v.size() ) return v[idx_in++];
      else return {};
    },
    grppi::keep(
      [this](int x) {
        invocations_op++;
        return x % 2 == 0; 
      }),
    [this](int x) {
      invocations_out++;
      w[idx_out++] = x;
    });
  }

  template <typename E>
  void run_discard_single(const E & e) {
    grppi::pipeline (e,
    [this]() -> optional<int> {
      invocations_in++;
      if(idx_in < v.size() ) return v[idx_in++];
      else return {};
    },
    grppi::discard(
      [this](int x) {
        invocations_op++;
        return x % 2 == 0; 
      }),
    [this](int x) {
      invocations_out++;
      w[idx_out++] = x;
    });
  }

  void check_keep_single() {
    EXPECT_EQ(2, invocations_in); // two invocation
    EXPECT_EQ(1, invocations_op); // one invocation
    EXPECT_EQ(1, invocations_out); // one invocation
    EXPECT_EQ(42, w[0]);
  }

  void check_discard_single() {
    EXPECT_EQ(2, invocations_in); // two invocation
    EXPECT_EQ(1, invocations_op); // one invocation
    EXPECT_EQ(0, invocations_out); // not invoked
    EXPECT_EQ(99, w[0]);
  }

  void setup_multiple() {
    v = vector<int>{1,2,3,4,5};
    w = vector<int>(5);
    expected_even = vector<int>{2,4};    
    expected_odd = vector<int>{1,3,5};
  }

  template <typename E>
  void run_keep_multiple(const E & e) {
    grppi::pipeline(e,
      [this]() -> optional<int> {
        invocations_in++;
        if (idx_in < v.size() ) return v[idx_in++];
        else return {};
      },
      grppi::keep(
        [this](int x) {
        invocations_op++;
        return x % 2 == 0;
      }),
      [this](int x) {
        invocations_out++;
        w[this->idx_out++] = x;
      });
  }

  template <typename E>
  void run_discard_multiple(const E & e) {
    grppi::pipeline(e,
      [this]() -> optional<int> {
        invocations_in++;
        if (idx_in < v.size()) return v[idx_in++];
        else return {};
      },
      grppi::discard(
        [this](int x) {
        invocations_op++;
        return x % 2 == 0;
      }),
      [this](int x) {
        invocations_out++;
        w[idx_out++] = x;
      });
  }

  void check_keep_multiple() {
    EXPECT_EQ(6, invocations_in); // six invocations
    EXPECT_EQ(5, invocations_op); // five invocations
    EXPECT_EQ(2, invocations_out); // two invocations
    EXPECT_TRUE(equal(begin(expected_even), end(expected_even), begin(w)));
  }

  void check_discard_multiple() {
    EXPECT_EQ(6, invocations_in); // six invocations
    EXPECT_EQ(5, invocations_op); // five invocations
    EXPECT_EQ(3, invocations_out); // two invocations
    EXPECT_TRUE(equal(begin(expected_odd), end(expected_odd), begin(w)));
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(stream_filter_test, executions);

TYPED_TEST(stream_filter_test, static_ordered_keep_empty)
{
  this->setup_empty();
  this->execution_.enable_ordering();
  this->run_keep_empty(this->execution_);
  this->check_keep_empty();
}

TYPED_TEST(stream_filter_test, static_unordered_keep_empty)
{
  this->setup_empty();
  this->execution_.disable_ordering();
  this->run_keep_empty(this->execution_);
  this->check_keep_empty();
}

TYPED_TEST(stream_filter_test, dyn_keep_empty)
{
  this->setup_empty();
  this->run_keep_empty(this->dyn_execution_);
  this->check_keep_empty();
}

TYPED_TEST(stream_filter_test, static_discard_empty)
{
  this->setup_empty();
  this->run_discard_empty(this->execution_);
  this->check_discard_empty();
}

TYPED_TEST(stream_filter_test, dyn_discard_empty)
{
  this->setup_empty();
  this->run_discard_empty(this->dyn_execution_);
  this->check_discard_empty();
}

TYPED_TEST(stream_filter_test, static_keep_single)
{
  this->setup_single();
  this->run_keep_single(this->execution_);
  this->check_keep_single();
}

TYPED_TEST(stream_filter_test, dyn_keep_single)
{
  this->setup_single();
  this->run_keep_single(this->dyn_execution_);
  this->check_keep_single();
}

TYPED_TEST(stream_filter_test, static_discard_single)
{
  this->setup_single();
  this->run_discard_single(this->execution_);
  this->check_discard_single();
}

TYPED_TEST(stream_filter_test, dyn_discard_single)
{
  this->setup_single();
  this->run_discard_single(this->dyn_execution_);
  this->check_discard_single();
}

TYPED_TEST(stream_filter_test, static_keep_multiple)
{
  this->setup_multiple();
  this->run_keep_multiple(this->execution_);
  this->check_keep_multiple();
}

TYPED_TEST(stream_filter_test, dyn_keep_multiple)
{
  this->setup_multiple();
  this->run_keep_multiple(this->dyn_execution_);
  this->check_keep_multiple();
}

TYPED_TEST(stream_filter_test, static_discard_multiple)
{
  this->setup_multiple();
  this->run_discard_multiple(this->execution_);
  this->check_discard_multiple();
}

TYPED_TEST(stream_filter_test, dyn_discard_multiple)
{
  this->setup_multiple();
  this->run_discard_multiple(this->dyn_execution_);
  this->check_discard_multiple();
}
