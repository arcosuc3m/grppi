/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <experimental/optional>

#include <gtest/gtest.h>

#include "grppi/common/patterns.h"
#include "grppi/farm.h"
#include "grppi/stream_filter.h"
#include "grppi/pipeline.h"

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
