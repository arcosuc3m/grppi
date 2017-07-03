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
#include <utility>

#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include "common/mpmc_queue.h"

using namespace std;
using namespace grppi;

TEST(mpmc_queue, constructor){
  mpmc_queue<int> queue(10, queue_mode::lockfree);
  EXPECT_EQ(true,queue.is_empty());
}


TEST(mpmc_queue, lockfree_push_pop){
  mpmc_queue<int> queue(10, queue_mode::lockfree);
  auto inserted = queue.push(1);
  EXPECT_EQ(true, inserted);
  EXPECT_EQ(false, queue.is_empty());
  auto value = queue.pop();
  EXPECT_EQ(true, queue.is_empty());
  EXPECT_EQ(1,value);
}

TEST(mpmc_queue, concurrent_push_pop){
mpmc_queue<int> queue(3, queue_mode::lockfree);
 std::vector<std::thread> thrs;
 for(auto i = 0; i<6; i++){
   thrs.push_back(std::thread([&,i](){queue.push(i);} ));
 } 
 auto val = 0;
 for(auto i = 0;i<6; i++){
   val += queue.pop();
 }
 for(auto & t : thrs) t.join();
 EXPECT_EQ(15, val);
 EXPECT_EQ(true, queue.is_empty());
}




