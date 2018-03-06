/**
* @version    GrPPI v0.3
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
#include <iostream>
#include <numeric>

#include <gtest/gtest.h>

#include "stencil.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class stencil_test : public ::testing::Test {
public:
  T execution_;
  dynamic_execution dyn_execution_{execution_};

  // Vectors
  vector<int> v{};
  vector<int> v2{};
  vector<int> w{};

  // Invocation counter
  std::atomic<int> invocations_operation{0};
  std::atomic<int> invocations_neighbour{0};

  // Stencil on a single sequence.
  // Every v[i] is assigned v[i] + v[i+1] or v[i] + 0 if v[i+1] does not exist
  template <typename E>
  void run_unary(E & ex) {
    grppi::stencil(ex, begin(v), end(v), begin(w),
      [this](auto it, auto n) { 
        invocations_operation++; 
        return *it + n;
      },
      [&](auto it) { 
        invocations_neighbour++;
        if (it+1 != v.end()) {
          return *(it+1);
        }
        else {
          return 0; 
        }
      }
    );
  }

  // Test an stencil on an empty sequence 
  void setup_empty() {
  }

  void check_empty() {
    ASSERT_EQ(0, invocations_operation); 
    ASSERT_EQ(0, invocations_neighbour); 
  }

  // Test an stencil on a single-element sequence
  void setup_single() {
    v = { 1 };
    w = { 1 };
  }

  void check_single() {
    EXPECT_EQ(1, invocations_operation); 
    EXPECT_EQ(1, invocations_neighbour); 
    EXPECT_EQ(1, w[0]);
  }

  // Test an stencil on a 5-elements sequence
  void setup_multiple() {
    v = { 1, 2, 3, 4, 5 };
    w = { 0, 0, 0, 0, 0 };
  }

  void check_multiple() {
    EXPECT_EQ(5, invocations_operation); 
    EXPECT_EQ(5, invocations_neighbour);
    vector<int>expected{ 3, 5, 7, 9, 5};
    EXPECT_TRUE(equal(begin(expected), end(expected), begin(w)));
  }

  // Stencil on two sequences.
  // Each v[i] is made 
  //     v[i-2] + v[i-1] + v[i+1] + v[i+2] +
  //     w[i-2] + w[i-1] + w[i+1] + w[i+2] +
  template <typename E>
  void run_nary_tuple_range(E &ex) {
    auto vec_surronding = [](auto first, auto last, auto it) {
      vector<int> result;
      if (distance(first,it)>=2) result.push_back(*prev(it,2));
      if (distance(first,it)>=1) result.push_back(*prev(it,1));
      if (distance(it,last)>1) result.push_back(*next(it,1));
      if (distance(it,last)>2) result.push_back(*next(it,2));
      return result;
    };

    grppi::stencil(ex, make_tuple(begin(v),begin(v2)), end(v), begin(w),
      // Stencil computes average of neighbours
      [this](auto, const auto & n) {
        invocations_operation++;
        return std::accumulate(begin(n),end(n),0);
      },
      // Neighbours are i-2, i-1, i+1, i+2 of currrent position
      [&,this](auto it, auto it2) {
        invocations_neighbour++;
        auto r1 = vec_surronding(begin(v), end(v), it);
        auto r2 = vec_surronding(begin(v2), end(v2), it2);
        r1.insert(end(r1), begin(r2), end(r2));
        return r1;
      }
    );
  }

  template <typename E>
  void run_nary_tuple_size(E & ex) {
    auto vec_surronding = [](auto first, auto last, auto it) {
      vector<int> result;
      if (distance(first,it)>=2) result.push_back(*prev(it,2));
      if (distance(first,it)>=1) result.push_back(*prev(it,1));
      if (distance(it,last)>1) result.push_back(*next(it,1));
      if (distance(it,last)>2) result.push_back(*next(it,2));
      return result;
    };

    grppi::stencil(ex, make_tuple(begin(v),begin(v2)), v.size(), begin(w),
      // Stencil computes average of neighbours
      [this](auto, const auto & n) {
        invocations_operation++;
        return std::accumulate(begin(n), end(n), 0);
      },
      // Neighbours are i-2, i-1, i+1, i+2 of currrent position
      [&,this](auto it, auto it2) {
        invocations_neighbour++;
        auto r1 = vec_surronding(begin(v), end(v), it);
        auto r2 = vec_surronding(begin(v2), end(v2), it2);
        r1.insert(end(r1), begin(r2), end(r2));
        return r1;
      }
    );
  }

  // Test an n-ary stencil on an empty sequence 
  void setup_empty_nary() {
  }

  void check_empty_nary() {
    ASSERT_EQ(0, invocations_operation);
    ASSERT_EQ(0, invocations_neighbour);
  }

  void setup_single_nary() {
    v = { 1 };
    v2 = { 1 };
    w = { 1 };
  }

  void check_single_nary() {
    EXPECT_EQ(1, invocations_operation); 
    EXPECT_EQ(1, invocations_neighbour); 
    EXPECT_EQ(0, w[0]);
  }

  void setup_multiple_nary() {
    generate_n(back_inserter(v), 10,
      [i=0]() mutable { return i++; });
    generate_n(back_inserter(v2), 10,
      [i=100]() mutable { return i++; });
    fill_n(back_inserter(w), 10, 0);

  }

  void check_multiple_nary() {
    EXPECT_EQ(10, invocations_operation);
    EXPECT_EQ(10, invocations_neighbour);

    vector<int> expected{
              1 + 2             + 101 + 102,
          0 + 2 + 3       + 100 + 102 + 103,
      0 + 1 + 3 + 4 + 100 + 101 + 103 + 104,
      1 + 2 + 4 + 5 + 101 + 102 + 104 + 105,
      2 + 3 + 5 + 6 + 102 + 103 + 105 + 106,
      3 + 4 + 6 + 7 + 103 + 104 + 106 + 107,
      4 + 5 + 7 + 8 + 104 + 105 + 107 + 108,
      5 + 6 + 8 + 9 + 105 + 106 + 108 + 109,
      6 + 7 + 9     + 106 + 107 + 109      ,
      7 + 8         + 107 + 108            
    };
    EXPECT_EQ(expected.size(), w.size());
    EXPECT_TRUE(equal(begin(expected), end(expected), begin(w), end(w)));
  }


};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(stencil_test, executions);

TYPED_TEST(stencil_test, static_empty)
{
  this->setup_empty();
  this->run_unary(this->execution_);
  this->check_empty();
}

TYPED_TEST(stencil_test, dyn_empty)
{
  this->setup_empty();
  this->run_unary(this->dyn_execution_);
  this->check_empty();
}

TYPED_TEST(stencil_test, static_empty_nary_tuple_range)
{
  this->setup_empty_nary();
  this->run_nary_tuple_range(this->execution_);
  this->check_empty_nary();
}

TYPED_TEST(stencil_test, dyn_empty_nary_tuple_range)
{
  this->setup_empty_nary();
  this->run_nary_tuple_range(this->dyn_execution_);
  this->check_empty_nary();
}

TYPED_TEST(stencil_test, static_empty_nary_tuple_size)
{
  this->setup_empty_nary();
  this->run_nary_tuple_size(this->execution_);
  this->check_empty_nary();
}

TYPED_TEST(stencil_test, dyn_empty_nary_tuple_size)
{
  this->setup_empty_nary();
  this->run_nary_tuple_size(this->dyn_execution_);
  this->check_empty_nary();
}

TYPED_TEST(stencil_test, static_single)
{
  this->setup_single();
  this->run_unary(this->execution_);
  this->check_single();
}

TYPED_TEST(stencil_test, dyn_single)
{
  this->setup_single();
  this->run_unary(this->dyn_execution_);
  this->check_single();
}

TYPED_TEST(stencil_test, static_single_nary_tuple_range)
{
  this->setup_single_nary();
  this->run_nary_tuple_range(this->execution_);
  this->check_single_nary();
}

TYPED_TEST(stencil_test, dyn_single_nary_tuple_range)
{
  this->setup_single_nary();
  this->run_nary_tuple_range(this->dyn_execution_);
  this->check_single_nary();
}

TYPED_TEST(stencil_test, static_single_nary_tuple_size)
{
  this->setup_single_nary();
  this->run_nary_tuple_size(this->execution_);
  this->check_single_nary();
}

TYPED_TEST(stencil_test, dyn_single_nary_tuple_size)
{
  this->setup_single_nary();
  this->run_nary_tuple_size(this->dyn_execution_);
  this->check_single_nary();
}

TYPED_TEST(stencil_test, static_multiple)
{
  this->setup_multiple();
  this->run_unary(this->execution_);
  this->check_multiple();
}

TYPED_TEST(stencil_test, dyn_multiple)
{
  this->setup_multiple();
  this->run_unary(this->dyn_execution_);
  this->check_multiple();
}

TYPED_TEST(stencil_test, static_multiple_nary_tuple_range)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_range(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(stencil_test, dyn_multiple_nary_tuple_range)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_range(this->dyn_execution_);
  this->check_multiple_nary();
}

TYPED_TEST(stencil_test, static_multiple_nary_tuple_size)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_size(this->execution_);
  this->check_multiple_nary();
}

TYPED_TEST(stencil_test, dyn_multiple_nary_tuple_size)
{
  this->setup_multiple_nary();
  this->run_nary_tuple_size(this->dyn_execution_);
  this->check_multiple_nary();
}
