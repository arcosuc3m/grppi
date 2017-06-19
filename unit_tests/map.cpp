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
#include <vector>

#include <gtest/gtest.h>

#include "map.h"
#include "common/polymorphic_execution.h"

using namespace std;

TEST(map, next_value)
{
  auto e = grppi::make_polymorphic_execution<grppi::sequential_execution>();
  vector<int> v{1,2,3,4,5};
  vector<int> w(5);
  grppi::map(e, begin(v), end(v), begin(w), [](int x) { return x+1; });

  vector<int> res{2, 3, 4, 5, 6};
  EXPECT_EQ(res, w);
}

TEST(map, add2_vec)
{
  auto e = grppi::make_polymorphic_execution<grppi::sequential_execution>();
  vector<int> v{1,2,3,4,5};
  vector<int> w{2,4,6,8,10};
  vector<int> r(5);
  grppi::map(e, begin(v), end(v), begin(r), [](int x, int y) { return x+y; }, begin(w));

  vector<int> res{3, 6, 9, 12, 15};
  EXPECT_EQ(res, r);
}

TEST(map, add3_vec)
{
  auto e = grppi::make_polymorphic_execution<grppi::sequential_execution>();
  vector<int> v{1,2,3,4,5};
  vector<int> w1{2,4,6,8,10};
  vector<int> w2{10, 20, 30, 40, 50};
  vector<int> r(5);
  grppi::map(e, begin(v), end(v), begin(r), 
             [](int x, int y, int z) { return x+y+z; },
             begin(w1), begin(w2));

  vector<int> res{13, 26, 39, 52, 65};
  EXPECT_EQ(res, r);
}
