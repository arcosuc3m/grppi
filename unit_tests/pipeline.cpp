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

#include "pipeline.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class pipeline_test : public ::testing::Test {
public:
  T execution_;

  // Variables
  int out;
  int expected;
  int counter;

  // Vectors
  vector<int> v{};

  // Invocation counter
  std::atomic<int> invocations_init{0};
  std::atomic<int> invocations_last{0};
  std::atomic<int> invocations_intermediate{0};

  void setup_two_stages_empty() {
  }

  void check_two_stages_empty() {
    ASSERT_EQ(1, invocations_init); 
    ASSERT_EQ(0, invocations_last); 
  }

  void setup_two_stages() {
    out = 0;
    counter = 2;
  }

  void check_two_stages() {
    ASSERT_EQ(2, invocations_init); 
    ASSERT_EQ(1, invocations_last); 
    EXPECT_EQ(1, this->out);
  }

  void setup_three_stages() {
    counter = 5;
    out = 0;
  }

  void check_three_stages() {
    ASSERT_EQ(5, invocations_init); 
    ASSERT_EQ(4, invocations_last); 
    ASSERT_EQ(4, invocations_intermediate);
    EXPECT_EQ(20, this->out);
  }


};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(pipeline_test, executions);

TYPED_TEST(pipeline_test, static_two_stages_empty)
{
  this->setup_two_stages_empty();
    grppi::pipeline( this->execution_,
    [this]() { 
        this->invocations_init++;
        return optional<int>(); 
    },
    [this]( auto x ) {
      this->invocations_last++;
    }
  );

  this->check_two_stages_empty();
}



TYPED_TEST(pipeline_test, static_two_stages)
{
  this->setup_two_stages();
    grppi::pipeline( this->execution_,
    [this]() { 
        this->invocations_init++;
        this->counter--;
        if(this->counter  == 0){
          return optional<int>(); 
        }else{
          return optional<int>(this->counter);
        }
    },
    [this]( auto x ) {
      this->invocations_last++;
      this->out += x;
    }
  );
  this->check_two_stages();
}



TYPED_TEST(pipeline_test, static_three_stages)
{
  this->setup_three_stages();
    grppi::pipeline( this->execution_,
    [this]() { 
        this->invocations_init++;
        this->counter--;
        if(this->counter  == 0){
          return optional<int>(); 
        }else{
          return optional<int>(this->counter);
        }
    },
    [this]( auto x ) {
      this->invocations_intermediate++;
      return x*2;
    },
    [this]( auto y ) {
      this->invocations_last++;
      this->out += y;
    }
  );
  this->check_three_stages();
}
