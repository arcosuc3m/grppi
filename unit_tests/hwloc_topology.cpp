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
#include <gtest/gtest.h>
#define private public
#include "common/hwloc_topology.h"
#include "common/native_topology.h"

using namespace std;

template <typename T>
class topology_test : public ::testing::Test {
};

using topologies = ::testing::Types<grppi::native_topology
#ifdef GRPPI_HWLOC
,grppi::hwloc_topology
#endif
>;

TYPED_TEST_CASE(topology_test, topologies);

TYPED_TEST(topology_test, properties)
{
  TypeParam t;
  EXPECT_TRUE(t.numa_nodes() > 0);
  EXPECT_TRUE(t.core_nodes() > 0);
  EXPECT_TRUE(t.logical_core_nodes() > 0);
}

TYPED_TEST(topology_test, iteration)
{
  TypeParam t;
  auto node = t.first_processor_unit();
  EXPECT_TRUE(node.index() == 0);
  node++;
  EXPECT_TRUE(node.index() == 1);
}

// Private members tests

TEST(native_processor_unit_test, construct)
{
  grppi::native_processor_unit pu{0,2};
  EXPECT_EQ(0, pu.index());
  EXPECT_EQ(0, pu.os_index());
}

TEST(native_processor_unit_test, iterate)
{
  grppi::native_processor_unit pu{0,2};
  pu++;
  EXPECT_EQ(1, pu.index());
  EXPECT_EQ(1, pu.os_index());
  pu++;
  EXPECT_EQ(0, pu.index());
  EXPECT_EQ(0, pu.os_index());
}
