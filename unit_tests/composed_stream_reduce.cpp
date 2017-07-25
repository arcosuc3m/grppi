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

#include "pipeline.h"
#include "stream_reduce.h"
#include "poly/polymorphic_execution.h"


using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class composed_reduce_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

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
  std::atomic<int> invocations_cons{0};
  std::atomic<int> invocations_stage{0};

  void setup_window_offset() {
    out = 0;
    v = vector<int>{1,2,3,4,5,6};
    window = 2;
    offset = 1;
  }

  void check_window_offset() {
    EXPECT_EQ(7, invocations_gen);
    EXPECT_EQ(6, invocations_stage);
    EXPECT_EQ(25, invocations_reduce);
    EXPECT_EQ(5, invocations_cons);
    EXPECT_EQ(45, this->out);
  }

  void setup_offset_window() {
    out = 0;
    v = vector<int>{1,2,3,4,5,6,7,8,9,10};
    window = 2;
    offset = 4;
  }

  void check_offset_window() {
    EXPECT_EQ(11, invocations_gen);
    EXPECT_EQ(10, invocations_stage);
    EXPECT_EQ(15, invocations_reduce);
    EXPECT_EQ(3, invocations_cons);
    EXPECT_EQ(39, this->out);
  }
};

// Test for execution policies defined in supported_executions.h
using executions = ::testing::Types<
  grppi::parallel_execution_native,
  grppi::parallel_execution_omp
>;

TYPED_TEST_CASE(composed_reduce_test, executions);

TYPED_TEST(composed_reduce_test, static_window_offset)
{
  this->setup_window_offset();
  grppi::pipeline( this->execution_,
    [this]() -> optional<int>{
      this->invocations_gen++;
      if(this-> v.size() > 0) {
        auto problem = this->v.back();
        this->v.pop_back();
        return problem;
      }
      else {
        return {};
      }
    },
    [this](int a){
      this->invocations_stage++;
       return a+1;
    },
    grppi::stream_reduce(this->execution_,
      this->window, this->offset, 0,
      [this](int a, int b){
        this->invocations_reduce++;
        return a+b;
      }
    ),
    [this](int a){
      this->invocations_cons++;
      this-> out += a;
    }
  );
  this->check_window_offset();
}

TYPED_TEST(composed_reduce_test, static_offset_window)
{
  this->setup_offset_window();
  grppi::pipeline( this->execution_,
    [this]() -> optional<int>{
      this->invocations_gen++;
      if(this-> v.size() > 0) {
        auto problem = this->v.back();
        this->v.pop_back();
        return problem;
      }
      else {
        return {};
      }
    },
    [this](int a){
      this->invocations_stage++;
       return a+1;
    },
    grppi::stream_reduce(this->execution_,
      this->window, this->offset, 0,
      [this](int a, int b){
        this->invocations_reduce++;
        return a+b;
      }
    ),
    [this](int a){
      this->invocations_cons++;
      this-> out += a;
    }
  );
  this->check_offset_window();
}




