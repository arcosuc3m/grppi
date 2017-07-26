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
#include <iostream>

#include <gtest/gtest.h>

#include "stencil.h"
#include "poly/polymorphic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class stencil_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Vectors
  vector<int> v{};
  vector<int> v2{};
  vector<int> w{};
  vector<int> expected{};
  int rowsize;

  // Invocation counter
  std::atomic<int> invocations_operation{0};
  std::atomic<int> invocations_neighbour{0};

  void setup_empty_ary() {
  }

  void check_empty_ary() {
    ASSERT_EQ(0, invocations_operation);
    ASSERT_EQ(0, invocations_neighbour);
  }


  void setup_empty() {
  }

  void check_empty() {
    ASSERT_EQ(0, invocations_operation); 
    ASSERT_EQ(0, invocations_neighbour); 
  }

  void setup_single() {
    v = vector<int>{1};
    w = vector<int>{1};
  }

  void check_single() {
    EXPECT_EQ(1, invocations_operation); 
    EXPECT_EQ(1, invocations_neighbour); 
    EXPECT_EQ(1, this->w[0]);
  }

  void setup_single_ary() {
    rowsize=1;
    v = vector<int>{1};
    v2 = vector<int>{1};
    w = vector<int>{1};
  }

  void check_single_ary() {
    EXPECT_EQ(1, invocations_operation); 
    EXPECT_EQ(1, invocations_neighbour); 
    EXPECT_EQ(2, this->w[0]);
  }

  void setup_multiple_ary() {
    rowsize= 3;
    v = vector<int>{1,2,3,
                    4,5,6,
                    7,8,9};
    v2 = vector<int>{1,2,3,
                    4,5,6,
                    7,8,9};
    w = vector<int>{0,0,0,0,0,0,0,0,0};
    expected = vector<int>{7,9,11,
                           13,13,13,
                           15,17,18};
  }

  void check_multiple_ary() {
    EXPECT_EQ(9, invocations_operation);
    EXPECT_EQ(9, invocations_neighbour);
    EXPECT_TRUE(equal(begin(this->expected), end(this->expected), begin(this->w)));

  }


  void setup_multiple() {
    v = vector<int>{1,2,3,4,5};
    w = vector<int>{0,0,0,0,0};
    expected = vector<int>{3,5,7,9,5};

  }

  void check_multiple() {
    EXPECT_EQ(5, invocations_operation); 
    EXPECT_EQ(5, invocations_neighbour);
    EXPECT_TRUE(equal(begin(this->expected), end(this->expected), begin(this->w)));

  }
};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(stencil_test, executions);

TYPED_TEST(stencil_test, static_empty)
{
  this->setup_empty();
  grppi::stencil(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) { 
      this->invocations_operation++; 
      return 0;
    },
    [this](auto it) { 
      this->invocations_neighbour++; 
      return 0; 
    }
  );
  this->check_empty();
}

TYPED_TEST(stencil_test, poly_empty)
{
  this->setup_empty();
  grppi::stencil(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) { 
      this->invocations_operation++; 
      return 0;
    },
    [this](auto it) { 
      this->invocations_neighbour++; 
      return 0; 
    }
  );
  this->check_empty();
}

TYPED_TEST(stencil_test, static_single)
{
  this->setup_single();
  grppi::stencil(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) { 
      this->invocations_operation++; 
      return *(it) + x;
    },
    [&](auto it) { 
      this->invocations_neighbour++;
      if( it+1 != this->v.end() ){
        return *(it+1);
      }else{
        return 0; 
      }
    }
  );
  this->check_single();
}

TYPED_TEST(stencil_test, poly_single)
{
  this->setup_single();
  grppi::stencil(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) { 
      this->invocations_operation++; 
      return *(it) + x;
    },
    [&](auto it) { 
      this->invocations_neighbour++;
      if( it+1 != this->v.end() ){
        return *(it+1);
      }else{
        return 0; 
      }
    }
  );
  this->check_single();
}


TYPED_TEST(stencil_test, static_multiple)
{
  this->setup_multiple();
  grppi::stencil(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) { 
      this->invocations_operation++; 
      return *(it) + x;
    },
    [&](auto it) { 
      this->invocations_neighbour++;
      if( it+1 != this->v.end() ){
        return *(it+1);
      }else{
        return 0; 
      }
    }
  );
  this->check_multiple();
}

TYPED_TEST(stencil_test, poly_multiple)
{
  this->setup_multiple();
  grppi::stencil(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) { 
      this->invocations_operation++; 
      return *(it) + x;
    },
    [&](auto it) { 
      this->invocations_neighbour++;
      if( it+1 != this->v.end() ){
        return *(it+1);
      }else{
        return 0; 
      }
    }
  );
  this->check_multiple();
}

TYPED_TEST(stencil_test, static_empty_ary)
{
  this->setup_empty_ary();
  grppi::stencil(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) {
      this->invocations_operation++;
      return 0;
    },
    [this](auto it,auto it2) {
      this->invocations_neighbour++;
      return 0;
    }, begin(this->v2)
  );
  this->check_empty_ary();
}

/*
TYPED_TEST(stencil_test, static_single_ary)
{
  this->setup_single_ary();
  grppi::stencil(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto ng) {
      this->invocations_operation++;
      auto val = 0;
      for(auto j = 0; j< ng[0].size(); j++) val += ng[0][j] + ng[1][j];
      return val/ng[0].size();
    },
    [&](auto it, auto it2) {
      this->invocations_neighbour++;
      std::vector<std::vector<int>> neighbours(2,std::vector<int>());
      for(auto i= 0; i < 4; ++i){
        auto pos = it + (i%this->rowsize) + (i/this->rowsize) + ((i/2)*this->rowsize);
        auto pos2 = it2 + (i%this->rowsize) + (i/this->rowsize)+((i/2)*this->rowsize);
        if(pos < this->v.end() ){
           neighbours[0].push_back(*pos);
           neighbours[1].push_back(*pos2);
        }
        return neighbours;
      }
    }, begin(this->v2)
  );
  this->check_single_ary();
}

TYPED_TEST(stencil_test, static_multiple_ary)
{
  this->setup_multiple_ary();
  grppi::stencil(this->execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto ng) {
      this->invocations_operation++;
      auto val = 0;
      for(auto j = 0; j< ng[0].size(); j++) val += ng[0][j] + ng[1][j];
      return val/ng[0].size();
    },
    [&](auto it, auto it2) {
      this->invocations_neighbour++;
      std::vector<std::vector<int>> neighbours(2,std::vector<int>());
      for(auto i= 0; i < 4; ++i){
        auto pos = it + (i%this->rowsize) + (i/this->rowsize) + ((i/2)*this->rowsize);
        auto pos2 = it2 + (i%this->rowsize) + (i/this->rowsize)+((i/2)*this->rowsize);
        if(pos < this->v.end() ){
           neighbours[0].push_back(*pos);
           neighbours[1].push_back(*pos2);
        }
      }
      return neighbours;
    }, begin(this->v2)
  );
  this->check_multiple_ary();
}

TYPED_TEST(stencil_test, poly_empty_ary)
{
  this->setup_empty_ary();
  grppi::stencil(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto x) {
      this->invocations_operation++;
      return 0;
    },
    [this](auto it,auto it2) {
      this->invocations_neighbour++;
      return 0;
    }, begin(this->v2)
  );
  this->check_empty_ary();
}

TYPED_TEST(stencil_test, poly_single_ary)
{
  this->setup_single_ary();
  grppi::stencil(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto ng) {
      this->invocations_operation++;
      auto val = 0;
      for(auto j = 0; j< ng[0].size(); j++) val += ng[0][j] + ng[1][j];
      return val/ng[0].size();
    },
    [&](auto it, auto it2) {
      this->invocations_neighbour++;
      std::vector<std::vector<int>> neighbours(2,std::vector<int>());
      for(auto i= 0; i < 4; ++i){
        auto pos = it + (i%this->rowsize) + (i/this->rowsize) + ((i/2)*this->rowsize);
        auto pos2 = it2 + (i%this->rowsize) + (i/this->rowsize)+((i/2)*this->rowsize);
        if(pos < this->v.end() ){
           neighbours[0].push_back(*pos);
           neighbours[1].push_back(*pos2);
        }
        return neighbours;
      }
    }, begin(this->v2)
  );
  this->check_single_ary();
}

TYPED_TEST(stencil_test, poly_multiple_ary)
{
  this->setup_multiple_ary();
  grppi::stencil(this->poly_execution_, begin(this->v), end(this->v), begin(this->w),
    [this](auto it, auto ng) {
      this->invocations_operation++;
      auto val = 0;
      for(auto j = 0; j< ng[0].size(); j++) val += ng[0][j] + ng[1][j];
      return val/ng[0].size();
    },
    [&](auto it, auto it2) {
      this->invocations_neighbour++;
      std::vector<std::vector<int>> neighbours(2,std::vector<int>());
      for(auto i= 0; i < 4; ++i){
        auto pos = it + (i%this->rowsize) + (i/this->rowsize) + ((i/2)*this->rowsize);
        auto pos2 = it2 + (i%this->rowsize) + (i/this->rowsize)+((i/2)*this->rowsize);
        if(pos < this->v.end() ){
           neighbours[0].push_back(*pos);
           neighbours[1].push_back(*pos2);
        }
      }
      return neighbours;
    }, begin(this->v2)
  );
  this->check_multiple_ary();
}
*/

