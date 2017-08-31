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

#include "farm.h"
#include "pipeline.h"
#include "stream_iteration.h"

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class stream_iteration_test : public ::testing::Test {
public:
  T execution_;

  polymorphic_execution poly_execution_ =
    make_polymorphic_execution<T>();

  // Variables
  int out;
  int count;
  int n;

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
      this->invocations_gen++;
      if (this->count < this->n) {
        this->count+=1;
        return 1;
      }
      else return {};
    },
    grppi::repeat_until(
      grppi::pipeline(
        [this](int val){
          this->invocations_stage1++;
          return val+1;
        },
        [this](int val){
          this->invocations_stage2++;
          return val+1;
        }),
        [this](int val) {
          this->invocations_pred++;
          return val>=10;
        }
    ),
    [this](int val) {
      this->invocations_cons++;
      this->out += val;
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

using executions = ::testing::Types<
  grppi::sequential_execution,
  grppi::parallel_execution_native
>;

TYPED_TEST_CASE(stream_iteration_test, executions);

TYPED_TEST(stream_iteration_test, static_no_composed)
{ 
  this->setup_no_composed();
  this->run_nested_iteration(this->execution_);
  this->check_no_composed();
}

TYPED_TEST(stream_iteration_test, poly_no_composed)
{ 
  this->setup_no_composed();
  this->run_nested_iteration(this->poly_execution_);
  this->check_no_composed();
}

TYPED_TEST(stream_iteration_test, static_composed_pipeline)
{
  this->setup_composed_pipeline();
  this->run_nested_iteration_pipeline(this->execution_);
  this->check_composed_pipeline();
}

TYPED_TEST(stream_iteration_test, poly_composed_pipeline)
{
  this->setup_composed_pipeline();
  this->run_nested_iteration_pipeline(this->poly_execution_);
  this->check_composed_pipeline();

}

/*
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

TYPED_TEST(stream_iteration_test, poly_composed_farm)
{
  this->setup_composed_farm();
  grppi::repeat_until(this->poly_execution_,
    [this]() -> optional<int> {
      this->invocations_gen++;
      if (this->count < this->n) {
       this->count+=1;
       return 1;
      }
      else return {};
    },
    grppi::farm(this->poly_execution_,
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

