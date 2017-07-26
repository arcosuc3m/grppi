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
#include <gtest/gtest.h>

#include "poly/polymorphic_execution.h"

using namespace std;
using namespace grppi;

TEST(poly_exec, empty)
{
  polymorphic_execution e;
  EXPECT_FALSE(e.has_execution());
  EXPECT_EQ(nullptr, e.execution_ptr<sequential_execution>());
  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_native>());
  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_omp>());
  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_tbb>());
}

TEST(poly_exec, sequential)
{
  auto e = make_polymorphic_execution<sequential_execution>();

  EXPECT_TRUE(e.has_execution());

  EXPECT_NE(nullptr, e.execution_ptr<sequential_execution>());
  EXPECT_EQ(typeid(sequential_execution), e.type());

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_native>());
  EXPECT_NE(typeid(parallel_execution_native), e.type());

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_omp>());
  EXPECT_NE(typeid(parallel_execution_omp), e.type());

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_tbb>());
  EXPECT_NE(typeid(parallel_execution_tbb), e.type());
}


TEST(poly_exec, parallel_thr)
{
  auto e = make_polymorphic_execution<parallel_execution_native>();
  EXPECT_TRUE(e.has_execution());

  EXPECT_EQ(nullptr, e.execution_ptr<sequential_execution>());
  EXPECT_NE(typeid(sequential_execution), e.type());

  EXPECT_NE(nullptr, e.execution_ptr<parallel_execution_native>());
  EXPECT_EQ(typeid(parallel_execution_native), e.type());

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_omp>());
  EXPECT_NE(typeid(parallel_execution_omp), e.type());

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_tbb>());
  EXPECT_NE(typeid(parallel_execution_tbb), e.type());
}

TEST(poly_exec, parallel_omp)
{
  auto e = make_polymorphic_execution<parallel_execution_omp>();

  if (is_supported<parallel_execution_omp>()) {
    EXPECT_TRUE(e.has_execution());
  }
  else {
    EXPECT_FALSE(e.has_execution());
  }

  EXPECT_EQ(nullptr, e.execution_ptr<sequential_execution>());
  EXPECT_FALSE(e.is_execution<sequential_execution>());

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_native>());
  EXPECT_FALSE(e.is_execution<parallel_execution_native>());

  if (is_supported<parallel_execution_omp>()) {
    EXPECT_NE(nullptr, e.execution_ptr<parallel_execution_omp>());
    EXPECT_TRUE(e.is_execution<parallel_execution_omp>());
  }
  else {
    EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_omp>());
    EXPECT_FALSE(e.is_execution<parallel_execution_omp>());
  }

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_tbb>());
  EXPECT_FALSE(e.is_execution<parallel_execution_tbb>());
}

TEST(poly_exec, parallel_tbb)
{
  auto e = make_polymorphic_execution<parallel_execution_tbb>();

  if (is_supported<parallel_execution_tbb>()) {
    EXPECT_TRUE(e.has_execution());
  }
  else {
    EXPECT_FALSE(e.has_execution());
  }

  EXPECT_EQ(nullptr, e.execution_ptr<sequential_execution>());
  EXPECT_FALSE(e.is_execution<sequential_execution>());;

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_native>());
  EXPECT_FALSE(e.is_execution<parallel_execution_native>());

  EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_omp>());
  EXPECT_FALSE(e.is_execution<parallel_execution_omp>());

  if (is_supported<parallel_execution_tbb>()) {
    EXPECT_NE(nullptr, e.execution_ptr<parallel_execution_tbb>());
    EXPECT_TRUE(e.is_execution<parallel_execution_tbb>());
  }
  else {
    EXPECT_EQ(nullptr, e.execution_ptr<parallel_execution_tbb>());
    EXPECT_FALSE(e.is_execution<parallel_execution_tbb>());
  }
}
