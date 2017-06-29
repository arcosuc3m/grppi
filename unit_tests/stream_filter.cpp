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

#include "stream_filter.h"
#include "pipeline.h"
#include "common/polymorphic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class stream_filter_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Vectors
  vector<int> v{};
  vector<int> w{};
  vector<int> expected{};

  // Entry counter
  int idx_in = 0;
  int idx_out = 0;
 
  // Invocation counter
  std::atomic<int> invocations_in{0};
  std::atomic<int> invocations_op{0};
  std::atomic<int> invocations_out{0};

  void setup_empty() {
  }

  void check_empty() {
    ASSERT_EQ(1, invocations_in); // Functor in was invoked only once
    ASSERT_EQ(0, invocations_op); // Functor op was not invoked
    ASSERT_EQ(0, invocations_out); // Functor out was not invoked
  }

  void setup_single() {
    v = vector<int>{42};
    w = vector<int>{99};
  }

  void check_single_filtered() {
    EXPECT_EQ(2, this->invocations_in); // two invocation
    EXPECT_EQ(1, this->invocations_op); // one invocation
    EXPECT_EQ(1, this->invocations_out); // one invocation
    EXPECT_EQ(42, this->w[0]);
  }

  void check_single_unfiltered() {
    EXPECT_EQ(2, this->invocations_in); // two invocation
    EXPECT_EQ(1, this->invocations_op); // one invocation
    EXPECT_EQ(0, this->invocations_out); // not invoked
    EXPECT_EQ(99, this->w[0]);
  }

  void setup_multiple() {
    v = vector<int>{1,2,3,4,5};
    w = vector<int>(5);
    expected = vector<int>{2,4};
  }

  void check_multiple() {
    EXPECT_EQ(6, this->invocations_in); // six invocations
    EXPECT_EQ(5, this->invocations_op); // five invocations
    EXPECT_EQ(2, this->invocations_out); // two invocations
    EXPECT_TRUE(equal(begin(this->expected), end(this->expected), begin(this->w)));
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(stream_filter_test, executions);

TYPED_TEST(stream_filter_test, static_empty)
{
  this->setup_empty();
  grppi::stream_filter(this->execution_,
    [this]() { 
      this->invocations_in++;
      return optional<int>{}; 
    },
    [this](int x) { 
      this->invocations_op++; 
      return true; 
    },
    [this](int x) { 
      this->invocations_out++; 
    }
  );
  this->check_empty();
}

TYPED_TEST(stream_filter_test, static_single_filtered)
{
  this->setup_single();
  grppi::stream_filter(this->execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    [this](int x) {
      this->invocations_op++;
      return x % 2 == 0; 
    },
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_single_filtered();
}

TYPED_TEST(stream_filter_test, static_single_unfiltered)
{
  this->setup_single();
  grppi::stream_filter(this->execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    [this](int x) {
      this->invocations_op++;
      return x % 2 != 0;
    },
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_single_unfiltered();
}


TYPED_TEST(stream_filter_test, static_multiple)
{
  this->setup_multiple();
  grppi::stream_filter(this->execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    [this](int x) {
      this->invocations_op++;
      return x % 2 == 0;
    },
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_multiple();
}


TYPED_TEST(stream_filter_test, poly_empty)
{
  this->setup_empty();
  grppi::stream_filter(this->poly_execution_,
    [this]() -> optional<int> {
      this->invocations_in++;
      return {};
    },
    [this](int x) {
      this->invocations_op++;
      return true;
    },
    [this](int x) {
      this->invocations_out++;
    }
  );
  this->check_empty();
}

TYPED_TEST(stream_filter_test, poly_single_filtered)
{
  this->setup_single();
  grppi::stream_filter(this->poly_execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    [this](int x) {
      this->invocations_op++;
      return x % 2 == 0;
    },
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_single_filtered();
}

TYPED_TEST(stream_filter_test, poly_single_unfiltered)
{
  this->setup_single();
  grppi::stream_filter(this->poly_execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    [this](int x) {
      this->invocations_op++;
      return x % 2 != 0;
    },
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_single_unfiltered();
}


TYPED_TEST(stream_filter_test, poly_multiple)
{
  this->setup_multiple();
  grppi::stream_filter(this->poly_execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    [this](int x) {
      this->invocations_op++;
      return x % 2 == 0;
    },
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_multiple();
}


TYPED_TEST(stream_filter_test, static_empty_composed)
{
  this->setup_empty();
  grppi::pipeline(this->execution_,
    [this]() {
      this->invocations_in++;
      return optional<int>{};
    },
    grppi::stream_filter(this->execution_,
      [this](int x) {
        this->invocations_op++;
        return true;
      }
    ),
    [this](int x) {
      this->invocations_out++;
    }
  );
  this->check_empty();
}

TYPED_TEST(stream_filter_test, static_single_filtered_composed)
{
  this->setup_single();
  grppi::pipeline(this->execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    grppi::stream_filter(this->execution_,
      [this](int x) {
        this->invocations_op++;
        return x % 2 == 0;
      }
    ),
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_single_filtered();
}

TYPED_TEST(stream_filter_test, static_single_unfiltered_composed)
{
  this->setup_single();
  grppi::pipeline(this->execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    grppi::stream_filter(this->execution_,
      [this](int x) {
        this->invocations_op++;
         
        return x % 2 != 0;
      }
    ),
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_single_unfiltered();
}

TYPED_TEST(stream_filter_test, static_multiple_composed)
{
  this->setup_multiple();
  grppi::pipeline(this->execution_,
    [this]() {
      this->invocations_in++;
      return ( this->idx_in < this->v.size() ) ? optional<int>{this->v[this->idx_in++]} : optional<int>{};
    },
    grppi::stream_filter(this->execution_,
      [this](int x) {
        this->invocations_op++;
        return x % 2 == 0;
      }
    ),
    [this](int x) {
      this->invocations_out++;
      this->w[this->idx_out++] = x;
    }
  );
  this->check_multiple();
}

