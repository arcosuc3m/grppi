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

#include <gtest/gtest.h>

#include "map.h"
#include "common/polymorphic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class map_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Vectors
  vector<int> v{};
  vector<int> v2{};
  vector<int> v3{};
  vector<int> w{};
  vector<int> expected{};

  // Invocation counter
  std::atomic<int> invocations{0};

  void setup_empty() {
  }

  void check_empty() {
    ASSERT_EQ(0, invocations); // Functor was not invoked
  }

  void setup_single() {
    v = vector<int>{42};
    w = vector<int>{99};
  }

  void check_single() {
    EXPECT_EQ(1, this->invocations); // one invocation
    EXPECT_EQ(84, this->w[0]);
  }

  void setup_single_ary() {
    v = vector<int>{11};
    v2 = vector<int>{22};
    v3 = vector<int>{33};
    w = vector<int>{99};
  }

  void check_single_ary() {
    EXPECT_EQ(1, this->invocations); // one invocation
    EXPECT_EQ(66, this->w[0]);
  }

  void setup_multiple() {
    v = vector<int>{1,2,3,4,5};
    w = vector<int>(5);
    expected = vector<int>{2,4,6,8,10};
  }

  void check_multiple() {
    EXPECT_EQ(5, this->invocations); // five invocations
    EXPECT_TRUE(equal(begin(this->expected), end(this->expected), begin(this->w)));
  }

  void setup_multiple_ary() {
    v = vector<int>{1,2,3,4,5};
    v2 = vector<int>{2,4,6,8,10};
    v3 = vector<int>{10,10,10,10,10};
    w = vector<int>(5);
    expected = vector<int>{13,16,19,22,25};
  }

  void check_multiple_ary() {
    EXPECT_EQ(5, this->invocations); // five invocations
    EXPECT_TRUE(equal(begin(this->expected), end(this->expected), begin(this->w)));
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(map_test, executions);

TYPED_TEST(map_test, static_empty)
{
  this->setup_empty();
  grppi::map(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x) { 
      this->invocations++; 
      return x; 
    }
  );
  this->check_empty();
}

TYPED_TEST(map_test, poly_empty)
{
  this->setup_empty();
  grppi::map(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x) { 
      this->invocations++; 
      return x; 
    }
  );
  this->check_empty();
}

TYPED_TEST(map_test, static_empty_ary)
{
  this->setup_empty();
  grppi::map(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x, int y, int z) { 
      this->invocations++; 
      return x+y+z; 
    },
    begin(this->v2), begin(this->v3)
  );
  this->check_empty();
}

TYPED_TEST(map_test, poly_empty_ary)
{
  this->setup_empty();
  grppi::map(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x, int y, int z) { 
      this->invocations++; 
      return x+y+z; 
    },
    begin(this->v2), begin(this->v3)
  );
  this->check_empty();
}

TYPED_TEST(map_test, static_single)
{
  this->setup_single();
  grppi::map(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int i) {
      this->invocations++; 
      return i*2; 
    }
  );
  this->check_single();
}

TYPED_TEST(map_test, poly_single)
{
  this->setup_single();
  grppi::map(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int i) {
      this->invocations++; 
      return i*2; 
    }
  );
  this->check_single();
}

TYPED_TEST(map_test, static_single_ary)
{
  this->setup_single_ary();
  grppi::map(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x, int y, int z) {
      this->invocations++; 
      return x+y+z; 
    },
    begin(this->v2), begin(this->v3)
  );
  this->check_single_ary();
}

TYPED_TEST(map_test, poly_single_ary)
{
  this->setup_single_ary();
  grppi::map(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x, int y, int z) {
      this->invocations++; 
      return x+y+z; 
    },
    begin(this->v2), begin(this->v3)
  );
  this->check_single_ary();
}

TYPED_TEST(map_test, static_multiple)
{
  this->setup_multiple();
  grppi::map(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int i) {
      this->invocations++; 
      return i*2; 
    }
  );
  this->check_multiple();
}

TYPED_TEST(map_test, poly_multiple)
{
  this->setup_multiple();
  grppi::map(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int i) {
      this->invocations++; 
      return i*2; 
    }
  );
  this->check_multiple();
}

TYPED_TEST(map_test, static_multiple_ary)
{
  this->setup_multiple_ary();
  grppi::map(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x, int y, int z) {
      this->invocations++; 
      return x+y+z; 
    },
    begin(this->v2), begin(this->v3)
  );
  this->check_multiple_ary();
}

TYPED_TEST(map_test, poly_multiple_ary)
{
  this->setup_multiple_ary();
  grppi::map(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](int x, int y, int z) {
      this->invocations++; 
      return x+y+z; 
    },
    begin(this->v2), begin(this->v3)
  );
  this->check_multiple_ary();
}
