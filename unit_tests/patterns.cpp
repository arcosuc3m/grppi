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
#include <experimental/optional>

#include <gtest/gtest.h>

#include "common/patterns.h"
#include "farm.h"
#include "stream_filter.h"
#include "pipeline.h"

#include <iostream>

using namespace std;
using namespace grppi;

TEST(patterns, is_farm)
{
  auto f = [](int x) { return x; };
  auto pattern = farm(4,f);
  EXPECT_TRUE(is_farm<decltype(pattern)>);
  EXPECT_TRUE(is_farm<decltype(pattern)&>);
}

TEST(patterns, is_filter)
{
  auto f = [](int x) { return x; };
  auto pattern = keep(f);
  EXPECT_TRUE(is_filter<decltype(pattern)>);
  EXPECT_TRUE(is_filter<decltype(pattern)&>);
}

TEST(patterns, is_pipeline)
{
  auto f = [](int x) { return x; };
  auto g = [](int x) { return x+1; };
  auto pattern = pipeline(f,g);
  EXPECT_TRUE(is_pipeline<decltype(pattern)>);
  EXPECT_TRUE(is_pipeline<decltype(pattern)&>);
}
