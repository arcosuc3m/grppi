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

#include "grppi/pipeline.h"
#include "grppi/stream_iteration.h"
#include "grppi/dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class stream_iteration_test : public ::testing::Test {
public:
  T execution_{};
  dynamic_execution dyn_execution_{execution_};

  // Variables
  int out{};
  int count{};
  int n{};

  // Invocation counter
  std::atomic<int> invocations_gen{0};
  std::atomic<int> invocations_cons{0};
  std::atomic<int> invocations_oper{0};
  std::atomic<int> invocations_pred{0};

  std::atomic<int> invocations_stage1{0};
  std::atomic<int> invocations_stage2{0};

  template <typename E>
  void run_nested_iteration(const E & e) {
    grppi::pipeline(e,
      [this]() -> optional<int> { 
        invocations_gen++; 
        if (count < n) {
          count++;;
          return 1;
        }
        else return {};
      },
      grppi::repeat_until(
        [this](int val){
          invocations_oper++; 
          return val+1;
        },
        [this](int val) { 
          invocations_pred++; 
          return val>=10;
        }),
      [this](int val) {
        invocations_cons++; 
        out += val;
      }
    );
  }

  template <typename E>
  void run_nested_iteration_pipeline(const E & e) {
  grppi::pipeline(e,
    [this]() -> optional<int> {
      invocations_gen++;
      if (count < n) {
        count+=1;
        return 1;
      }
      else return {};
    },
    grppi::repeat_until(
      grppi::pipeline(
        [this](int val){
          invocations_stage1++;
          return val+1;
        },
        [this](int val){
          invocations_stage2++;
          return val+1;
        }),
        [this](int val) {
          invocations_pred++;
          return val>=10;
        }
    ),
    [this](int val) {
      invocations_cons++;
      out += val;
    });
  }

  void setup_no_composed() {
    out = 0;
    n = 5;
    count=0;
  }

  void check_no_composed() {
    EXPECT_EQ(6, invocations_gen);
    EXPECT_EQ(45, invocations_oper);
    EXPECT_EQ(45, invocations_pred);
    EXPECT_EQ(5, invocations_cons);
    EXPECT_EQ(50, out);
  }

  void setup_composed_pipeline() {
    out = 0;
    n = 5;
    count=0;
  }

  void check_composed_pipeline() {
    EXPECT_EQ(6, invocations_gen);
    EXPECT_EQ(25, invocations_stage1);
    EXPECT_EQ(25, invocations_stage2);
    EXPECT_EQ(25, invocations_pred);
    EXPECT_EQ(5, invocations_cons);
    EXPECT_EQ(55, out);
  }

  void setup_composed_farm() {
    out = 0;
    n = 5;
    count=0;
  }

  void check_composed_farm() {
    EXPECT_EQ(6, invocations_gen);
    EXPECT_EQ(45, invocations_oper);
    EXPECT_EQ(45, invocations_pred);
    EXPECT_EQ(5, invocations_cons);
    EXPECT_EQ(50, out);
  }
};

TYPED_TEST_CASE(stream_iteration_test, executions);

TYPED_TEST(stream_iteration_test, static_no_composed)
{ 
  this->setup_no_composed();
  this->run_nested_iteration(this->execution_);
  this->check_no_composed();
}

TYPED_TEST(stream_iteration_test, dyn_no_composed)
{ 
  this->setup_no_composed();
  this->run_nested_iteration(this->dyn_execution_);
  this->check_no_composed();
}

/*
TYPED_TEST(stream_iteration_test, static_composed_pipeline)
{
  this->setup_composed_pipeline();
  this->run_nested_iteration_pipeline(this->execution_);
  this->check_composed_pipeline();
}

TYPED_TEST(stream_iteration_test, dyn_composed_pipeline)
{
  this->setup_composed_pipeline();
  this->run_nested_iteration_pipeline(this->dyn_execution_);
  this->check_composed_pipeline();

}

TYPED_TEST(stream_iteration_test, static_composed_farm)
{
  this->setup_composed_farm();
  grppi::repeat_until(this->execution_,
    [this]() -> optional<int> {
      this->invocations_gen++;
      if (this->count < this->n) {
       this->count+=1;
       return 1;
      }
      else return {};
    },
    grppi::farm(this->execution_,
       [this](int val){
         this->invocations_oper++;
         return val+1;
       }
    ),
    [this](int val) {
      this->invocations_pred++;
      return val>=10;
    },
    [this](int val) {
      this->invocations_cons++;
      this->out += val;
    }
  );
  this->check_composed_farm();
}

TYPED_TEST(stream_iteration_test, dyn_composed_farm)
{
  this->setup_composed_farm();
  grppi::repeat_until(this->dyn_execution_,
    [this]() -> optional<int> {
      this->invocations_gen++;
      if (this->count < this->n) {
       this->count+=1;
       return 1;
      }
      else return {};
    },
    grppi::farm(this->dyn_execution_,
       [this](int val){
         this->invocations_oper++;
         return val+1;
       }
    ),
    [this](int val) {
      this->invocations_pred++;
      return val>=10;
    },
    [this](int val) {
      this->invocations_cons++;
      this->out += val;
    }
  );
  this->check_composed_farm();
}
*/

