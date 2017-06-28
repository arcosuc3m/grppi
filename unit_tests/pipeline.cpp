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
#include "farm.h"

#include "supported_executions.h"
#include "common/polymorphic_execution.h"

#include <iostream>
#include <numeric>

using namespace std;
using namespace grppi;

template <typename T>
class pipeline_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Variables
  int out;
  int counter;

  // Vectors
  vector<int> v{};
  vector<vector<int> > v2{};

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

  void setup_composed() {
    counter = 5;
    out = 0;

  }

  void check_composed() {
    ASSERT_EQ(5, invocations_init); 
    ASSERT_EQ(4, invocations_last); 
    ASSERT_EQ(4, invocations_intermediate);
    EXPECT_EQ(40, this->out);
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

TYPED_TEST(pipeline_test, poly_two_stages_empty)
{
  this->setup_two_stages_empty();
    grppi::pipeline( this->poly_execution_,
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

TYPED_TEST(pipeline_test, poly_two_stages)
{
  this->setup_two_stages();
    grppi::pipeline( this->poly_execution_,
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

TYPED_TEST(pipeline_test, poly_three_stages)
{
  this->setup_three_stages();
    grppi::pipeline( this->poly_execution_,
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



TYPED_TEST(pipeline_test, static_three_stages_composed)
{
  this->setup_composed();
  auto f_object =     grppi::farm(this->execution_,
        [this](std::vector<int> v) {
          this->invocations_intermediate++;
          int acumm = 0; 
          for(int i = 0; i < v.size(); i++ ){
            acumm += v[i];
          }
          return acumm;
        }
    );

    grppi::pipeline( this->execution_,
    [this]() { 
        this->invocations_init++;
        this->counter--;
        std::vector<int> v(5);
        std::iota(begin(v), end(v), 0);

        if(this->counter  <= 0){
          return optional< std::vector<int> >();
        }else{
          return optional<std::vector<int>>(v);
        }
    },
    f_object,
    [this]( auto y ) {
      this->invocations_last++;
      this->out += y;
    }
  );

  this->check_composed();
}
