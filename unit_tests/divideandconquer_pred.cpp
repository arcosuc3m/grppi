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
class divideconquer_test_pred : public ::testing::Test {
public:
  T execution_;
  grppi::dynamic_execution dyn_execution_{execution_};

  // Variables
  int out;

  // Vectors
  vector<int> v{};

  // Invocation counter
  std::atomic<int> invocations_divide{0};
  std::atomic<int> invocations_predicate{0};
  std::atomic<int> invocations_merge{0};
  std::atomic<int> invocations_base{0};

  // run simple with predicate
  template <typename E>
  auto run_simple_pred(const E & e) {
    return grppi::divide_conquer(e, v,
      // Divide
      [this](auto & v) {
        invocations_divide++;
        return std::vector<std::vector<int> >{};
      },
	  // Predicate
	  [this](auto x) {
    	  invocations_predicate++;
    	  return true;
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

  // run_vecsum with predicate
  template <typename E>
    auto run_vecsum_pred(const E & e) {
      return grppi::divide_conquer(e, v,
        // Divide
        [this](auto & v) {
          invocations_divide++;
          std::vector<std::vector<int>> subproblem;
          //if (v.size()>2) {
            auto mid = std::next(v.begin(), v.size()/2);
            subproblem.push_back({v.begin(), mid});
            subproblem.push_back({mid, v.end()});
          //}
          return subproblem;
        },
		// predicate
		[this](auto x) {
        	invocations_predicate++;
        	return x.size()>2;
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

  // run_vecsum_chunked_pred
  template <typename E>
    auto run_vecsum_chunked_pred(const E & e) {
      return grppi::divide_conquer(e, v,
        // Divide
        [this](auto & v) {
          invocations_divide++;
          std::vector<std::vector<int>> subproblem;
          //if (v.size()>3) {
            auto mid1 = std::next(v.begin(), v.size()/3);
            auto mid2 = std::next(v.begin(), 2*v.size()/3);
            subproblem.push_back({v.begin(), mid1});
            subproblem.push_back({mid1,mid2});
            subproblem.push_back({mid2, v.end()});
          //}
          return subproblem;
        },
		// predicate
		[this](auto x) {
        	invocations_predicate++;
        	return x.size()>3;
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

  void setup_empty_pred() {
  }

  void check_empty_pred() {
    ASSERT_EQ(0, invocations_divide);
    ASSERT_EQ(1, invocations_predicate); // predicate functor called only once
    ASSERT_EQ(1, invocations_base);
    ASSERT_EQ(0, invocations_merge);
  }

  void setup_single_pred() {
    out = 0;
    v = vector<int>{1}; // one element
  }

  void check_single_pred() {
	  EXPECT_EQ(0, this->invocations_divide);
	  EXPECT_EQ(1, this->invocations_predicate);
	  EXPECT_EQ(1, this->invocations_base);
	  EXPECT_EQ(0, this->invocations_merge);
	  EXPECT_EQ(0, this->out);
  }

  void setup_multiple_pred() {
    v = vector<int>{1,2,3,4,5,6,7,8,9,10};
    out = 0;
  }

  void check_multiple_pred() {
      EXPECT_EQ(0, this->invocations_divide);
      EXPECT_EQ(1, this->invocations_predicate);
      EXPECT_EQ(1, this->invocations_base);
      EXPECT_EQ(0, this->invocations_merge);
      EXPECT_EQ(55, this->out);
    }

  void setup_multiple_triple_div_pred() {
    v = vector<int>{1,2,3,4,5,6,7,8,9,10};
    out = 0;
  }

  void check_multiple_triple_div_pred() {
      EXPECT_EQ(0, this->invocations_divide);
      EXPECT_EQ(1, this->invocations_predicate);
      EXPECT_EQ(1, this->invocations_base);
      EXPECT_EQ(0, this->invocations_merge);
      EXPECT_EQ(55, this->out);
    }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(divideconquer_test_pred, executions);

TYPED_TEST(divideconquer_test_pred, static_empty_pred)
{
  this->setup_empty_pred();
  this->out = this->run_simple_pred(this->execution_);
  this->check_empty_pred();
}

TYPED_TEST(divideconquer_test_pred, dyn_empty_pred)
{
  this->setup_empty_pred();
  this->out = this->run_simple_pred(this->dyn_execution_);
  this->check_empty_pred();
}



TYPED_TEST(divideconquer_test_pred, static_single_pred)
{
  this->setup_single_pred();
  this->out = this->run_simple_pred(this->execution_);
  this->check_single_pred();
}

TYPED_TEST(divideconquer_test_pred, dyn_single_pred)
{
  this->setup_single_pred();
  this->out = this->run_simple_pred(this->dyn_execution_);
  this->check_single_pred();
}



TYPED_TEST(divideconquer_test_pred, static_multiple_pred)
{
  this->setup_multiple_pred();
  this->out =  this->run_vecsum_pred(this->execution_);
  this->check_multiple_pred();
}

TYPED_TEST(divideconquer_test_pred, dyn_multiple_pred)
{
  this->setup_multiple_pred();
  this->out =  this->run_vecsum_pred(this->dyn_execution_);
  this->check_multiple_pred();
}

TYPED_TEST(divideconquer_test_pred, static_multiple_single_thread_pred)
{
  this->setup_multiple_pred();
  this->execution_.set_concurrency_degree(1);
  this->out =  this->run_vecsum_pred(this->execution_);
  this->check_multiple_pred();
}

TYPED_TEST(divideconquer_test_pred, static_multiple_five_threads_pred)
{
  this->setup_multiple_pred();
  this->execution_.set_concurrency_degree(5);
  this->out =  this->run_vecsum_pred(this->execution_);
  this->check_multiple_pred();
}

TYPED_TEST(divideconquer_test_pred, static_multiple_triple_div_2_threads_pred)
{
  this->setup_multiple_triple_div_pred();
  this->execution_.set_concurrency_degree(2);
  this->out =  this->run_vecsum_chunked_pred(this->execution_);
  this->check_multiple_triple_div_pred();
}


TYPED_TEST(divideconquer_test_pred, static_multiple_triple_div_4_threads_pred)
{
  this->setup_multiple_triple_div_pred();
  this->execution_.set_concurrency_degree(4);
  this->out =  this->run_vecsum_chunked_pred(this->execution_);
  this->check_multiple_triple_div_pred();
}
