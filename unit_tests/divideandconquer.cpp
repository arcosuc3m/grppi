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

#include <gtest/gtest.h>

#include "divideandconquer.h"
#include "common/polymorphic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

template <typename T>
class divideandconquer_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Variables
  int out;

  // Vectors
  vector<int> v{};

  // Invocation counter
  std::atomic<int> invocations_divide{0};
  std::atomic<int> invocations_merge{0};
  std::atomic<int> invocations_base{0};

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
    EXPECT_EQ(1, this->out);
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
TYPED_TEST_CASE(divideandconquer_test, executions);

TYPED_TEST(divideandconquer_test, static_empty)
{
  this->setup_empty();
  this->out = grppi::divide_and_conquer(this->execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      return std::vector<std::vector<int> >{}; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++;
      return 0; 
    }, 
    // Combine
    [this](auto partial, auto out) { 
      this->invocations_merge++; 
    }
  );
  this->check_empty();
}

TYPED_TEST(divideandconquer_test, poly_empty)
{
  this->setup_empty();
  this->out = grppi::divide_and_conquer(this->poly_execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      return std::vector<std::vector<int> >{}; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++;
      return 0; 
    }, 
    // Combine
    [this](auto partial, auto out) { 
      this->invocations_merge++; 
    }
  );
  this->check_empty();
}



TYPED_TEST(divideandconquer_test, static_single)
{
  this->setup_single();
  this->out = grppi::divide_and_conquer(this->execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      return std::vector<std::vector<int>>{v}; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      return problem[0];
    }, 
    // Combine
    [this](auto partial, auto out) { 
      this->invocations_merge++; 
    }
  );
  this->check_single();
}

TYPED_TEST(divideandconquer_test, poly_single)
{
  this->setup_single();
  this->out = grppi::divide_and_conquer(this->poly_execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      return std::vector<std::vector<int>>{v}; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      return problem[0];
    }, 
    // Combine
    [this](auto partial, auto out) { 
      this->invocations_merge++; 
    }
  );
  this->check_single();
}



TYPED_TEST(divideandconquer_test, static_multiple)
{
  this->setup_multiple();
  
  this->out = grppi::divide_and_conquer(this->execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      std::vector<std::vector<int>> subproblem;
      if(v.size() > 2){
          std::vector<int> p1;
          std::vector<int> p2;
          int i=0;
          for(i; i < v.size()/2; i++){
              p1.push_back(v[i]);
          }
          for(i; i < v.size(); i++){
              p2.push_back(v[i]);
          }
          subproblem.push_back( p1);
          subproblem.push_back( p2);
      }
      return subproblem; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      int acumm = 0;
      for(int i=0; i < problem.size(); i++)
        acumm += problem[i];
      return acumm;
    }, 
    // Combine
    [this](auto & partial, auto & out) { 
      this->invocations_merge++; 
      out += partial;
    }
  );
  this->check_multiple();
}

TYPED_TEST(divideandconquer_test, poly_multiple)
{
  this->setup_multiple();
  
  this->out = grppi::divide_and_conquer(this->poly_execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      std::vector<std::vector<int>> subproblem;
      if(v.size() > 2){
          std::vector<int> p1;
          std::vector<int> p2;
          int i=0;
          for(i; i < v.size()/2; i++){
              p1.push_back(v[i]);
          }
          for(i; i < v.size(); i++){
              p2.push_back(v[i]);
          }
          subproblem.push_back( p1);
          subproblem.push_back( p2);
      }
      return subproblem; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      int acumm = 0;
      for(int i=0; i < problem.size(); i++)
        acumm += problem[i];
      return acumm;
    }, 
    // Combine
    [this](auto & partial, auto & out) { 
      this->invocations_merge++; 
      out += partial;
    }
  );
  this->check_multiple();
}



TYPED_TEST(divideandconquer_test, static_multiple_single_thread)
{
  this->setup_multiple();
  this->execution_.num_threads = 1;
  this->out = grppi::divide_and_conquer(this->execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      std::vector<std::vector<int>> subproblem;
      if(v.size() > 2){
          std::vector<int> p1;
          std::vector<int> p2;
          int i=0;
          for(i; i < v.size()/2; i++){
              p1.push_back(v[i]);
          }
          for(i; i < v.size(); i++){
              p2.push_back(v[i]);
          }
          subproblem.push_back( p1);
          subproblem.push_back( p2);
      }
      return subproblem; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      int acumm = 0;
      for(int i=0; i < problem.size(); i++)
        acumm += problem[i];
      return acumm;
    }, 
    // Combine
    [this](auto & partial, auto & out) { 
      this->invocations_merge++; 
      out += partial;
    }
  );
  this->check_multiple();
}



TYPED_TEST(divideandconquer_test, static_multiple_five_threads)
{
  this->setup_multiple();
  this->execution_.num_threads = 5;
  this->out = grppi::divide_and_conquer(this->execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      std::vector<std::vector<int>> subproblem;
      if(v.size() > 2){
          std::vector<int> p1;
          std::vector<int> p2;
          int i=0;
          for(i; i < v.size()/2; i++){
              p1.push_back(v[i]);
          }
          for(i; i < v.size(); i++){
              p2.push_back(v[i]);
          }
          subproblem.push_back( p1);
          subproblem.push_back( p2);
      }
      return subproblem; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      int acumm = 0;
      for(int i=0; i < problem.size(); i++)
        acumm += problem[i];
      return acumm;
    }, 
    // Combine
    [this](auto & partial, auto & out) { 
      this->invocations_merge++; 
      out += partial;
    }
  );
  this->check_multiple();
}



TYPED_TEST(divideandconquer_test, static_multiple_triple_div_2_threads)
{
  this->setup_multiple_triple_div();
  this->execution_.num_threads = 2;
  this->out = grppi::divide_and_conquer(this->execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      std::vector<std::vector<int>> subproblem;
      if (v.size() > 3){
          std::vector<int> p1;
          std::vector<int> p2;
          std::vector<int> p3;
          int i=0;
          for(i; i < v.size()/3; i++){
              p1.push_back(v[i]);
          }
          for(i; i < (v.size()*2)/3; i++){
              p2.push_back(v[i]);
          }
          for(i; i < v.size(); i++){
              p3.push_back(v[i]);
          }
          subproblem.push_back( p1);
          subproblem.push_back( p2);
          subproblem.push_back( p3);
      }
      return subproblem; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      int acumm = 0;
      for(int i=0; i < problem.size(); i++)
        acumm += problem[i];
      return acumm;
    }, 
    // Combine
    [this](auto & partial, auto & out) { 
      this->invocations_merge++; 
      out += partial;
    }
  );
  this->check_multiple_triple_div();
}


TYPED_TEST(divideandconquer_test, static_multiple_triple_div_4_threads)
{
  this->setup_multiple_triple_div();
  this->execution_.num_threads = 4;
  this->out = grppi::divide_and_conquer(this->execution_, this->v,
    // Divide
    [this](auto & v) { 
      this->invocations_divide++; 
      std::vector<std::vector<int>> subproblem;
      if (v.size() > 3){
          std::vector<int> p1;
          std::vector<int> p2;
          std::vector<int> p3;
          int i=0;
          for(i; i < v.size()/3; i++){
              p1.push_back(v[i]);
          }
          for(i; i < (v.size()*2)/3; i++){
              p2.push_back(v[i]);
          }
          for(i; i < v.size(); i++){
              p3.push_back(v[i]);
          }
          subproblem.push_back( p1);
          subproblem.push_back( p2);
          subproblem.push_back( p3);
      }
      return subproblem; 
    },
    // Solve base case
    [this](auto problem) { 
      this->invocations_base++; 
      int acumm = 0;
      for(int i=0; i < problem.size(); i++)
        acumm += problem[i];
      return acumm;
    }, 
    // Combine
    [this](auto & partial, auto & out) { 
      this->invocations_merge++; 
      out += partial;
    }
  );
  this->check_multiple_triple_div();
}
