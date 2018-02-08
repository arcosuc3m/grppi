/**
* @version    GrPPI v0.2
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
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
#include <numeric>

#include <gtest/gtest.h>

#include "divideconquer.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class divideconquer_test : public ::testing::Test {
public:
  T execution_;
  grppi::dynamic_execution dyn_execution_{execution_};

  // Variables
  int out;

  // Vectors
  vector<int> v{};

  // Invocation counter
  std::atomic<int> invocations_divide{0};
  std::atomic<int> invocations_merge{0};
  std::atomic<int> invocations_base{0};

  template <typename E>
  auto run_simple(const E & e) {
    return grppi::divide_conquer(e, v,
      // Divide
      [this](auto & v) { 
        invocations_divide++; 
        return std::vector<std::vector<int> >{}; 
      },
      // Solve base case
      [this](auto problem) { 
        invocations_base++;
        return 0; 
      }, 
      // Combine
      [this](auto out, auto partial) { 
        invocations_merge++;
        return 0; 
      });
  }

  template <typename E>
  auto run_vecsum(const E & e) {
    return grppi::divide_conquer(e, v,
      // Divide
      [this](auto & v) { 
        invocations_divide++; 
        std::vector<std::vector<int>> subproblem;
        if (v.size()>2) {
          auto mid = std::next(v.begin(), v.size()/2);
          subproblem.push_back({v.begin(), mid});
          subproblem.push_back({mid, v.end()});
        }
        return subproblem; 
      },
      // Solve base case
      [this](auto problem) { 
        invocations_base++; 
        return std::accumulate(problem.begin(), problem.end(), 0);
      }, 
      // Combine
      [this](auto  p1, auto  p2) { 
        invocations_merge++; 
        return p1 + p2;
      });
  }

  template <typename E>
  auto run_vecsum_chunked(const E & e) {
    return grppi::divide_conquer(e, v,
      // Divide
      [this](auto & v) { 
        invocations_divide++; 
        std::vector<std::vector<int>> subproblem;
        if (v.size()>3) {
          auto mid1 = std::next(v.begin(), v.size()/3);
          auto mid2 = std::next(v.begin(), 2*v.size()/3);
          subproblem.push_back({v.begin(), mid1});
          subproblem.push_back({mid1,mid2});
          subproblem.push_back({mid2, v.end()});
        }
        return subproblem; 
      },
      // Solve base case
      [this](auto problem) { 
        invocations_base++; 
        return std::accumulate(problem.begin(), problem.end(), 0);
      }, 
      // Combine
      [this](auto  p1, auto  p2) { 
        invocations_merge++; 
        return p1 + p2;
      });
  }
  void setup_empty() {
  }

  void check_empty() {
    ASSERT_EQ(1, invocations_divide); 
    ASSERT_EQ(1, invocations_base); 
    ASSERT_EQ(0, invocations_merge);
  }

  void setup_single() {
    out = 0;
    v = vector<int>{1};
  }

  void check_single() {
    EXPECT_EQ(1, this->invocations_divide);
    EXPECT_EQ(1, this->invocations_base); 
    EXPECT_EQ(0, this->invocations_merge);
    EXPECT_EQ(0, this->out);
  }

  void setup_multiple() {
    v = vector<int>{1,2,3,4,5,6,7,8,9,10};
    out = 0;
  }

  void check_multiple() {
    EXPECT_EQ(11, this->invocations_divide);
    EXPECT_EQ(6, this->invocations_base); 
    EXPECT_EQ(5, this->invocations_merge); 
    EXPECT_EQ(55, this->out);
  }

  void setup_multiple_triple_div() {
    v = vector<int>{1,2,3,4,5,6,7,8,9,10};
    out = 0;
  }

  void check_multiple_triple_div() {
    EXPECT_EQ(7, this->invocations_divide);
    EXPECT_EQ(5, this->invocations_base); 
    EXPECT_EQ(4, this->invocations_merge); 
    EXPECT_EQ(55, this->out);
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(divideconquer_test, executions_noff);

TYPED_TEST(divideconquer_test, static_empty)
{
  this->setup_empty();
  this->out = this->run_simple(this->execution_);
  this->check_empty();
}

TYPED_TEST(divideconquer_test, dyn_empty)
{
  this->setup_empty();
  this->out = this->run_simple(this->dyn_execution_);
  this->check_empty();
}



TYPED_TEST(divideconquer_test, static_single)
{
  this->setup_single();
  this->out = this->run_simple(this->execution_);
  this->check_single();
}

TYPED_TEST(divideconquer_test, dyn_single)
{
  this->setup_single();
  this->out = this->run_simple(this->dyn_execution_);
  this->check_single();
}



TYPED_TEST(divideconquer_test, static_multiple)
{
  this->setup_multiple();
  this->out =  this->run_vecsum(this->execution_);
  this->check_multiple();
}

TYPED_TEST(divideconquer_test, dyn_multiple)
{
  this->setup_multiple();
  this->out =  this->run_vecsum(this->dyn_execution_);
  this->check_multiple();
}

TYPED_TEST(divideconquer_test, static_multiple_single_thread)
{
  this->setup_multiple();
  this->execution_.set_concurrency_degree(1);
  this->out =  this->run_vecsum(this->execution_);
  this->check_multiple();
}

TYPED_TEST(divideconquer_test, static_multiple_five_threads)
{
  this->setup_multiple();
  this->execution_.set_concurrency_degree(5);
  this->out =  this->run_vecsum(this->execution_);
  this->check_multiple();
}

TYPED_TEST(divideconquer_test, static_multiple_triple_div_2_threads)
{
  this->setup_multiple_triple_div();
  this->execution_.set_concurrency_degree(2);
  this->out =  this->run_vecsum_chunked(this->execution_);
  this->check_multiple_triple_div();
}


TYPED_TEST(divideconquer_test, static_multiple_triple_div_4_threads)
{
  this->setup_multiple_triple_div();
  this->execution_.set_concurrency_degree(4);
  this->out =  this->run_vecsum_chunked(this->execution_);
  this->check_multiple_triple_div();
}
