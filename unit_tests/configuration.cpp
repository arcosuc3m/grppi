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
#include "common/configuration.h"

using namespace std;
using namespace grppi;

TEST(configuration_environment, get_config){
  configuration<> config;
  EXPECT_EQ(std::thread::hardware_concurrency(), config.concurrency_degree());
  EXPECT_TRUE(config.ordering());
  EXPECT_EQ(config.default_queue_size, config.queue_size());
  EXPECT_EQ(queue_mode::blocking, config.mode());
  EXPECT_STREQ("seq", config.dynamic_backend().c_str());
}

TEST(configuration_synthetic, get_correct_nthreads){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_NUM_THREADS") == 0) return "1";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(1, config.concurrency_degree());
}

TEST(configuration_synthetic, get_zero_nthreads){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_NUM_THREADS") == 0) return "0";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(std::thread::hardware_concurrency(), config.concurrency_degree());
}

TEST(configuration_synthetic, get_negative_nthreads){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_NUM_THREADS") == 0) return "-2";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(std::thread::hardware_concurrency(), config.concurrency_degree());
}

TEST(configuration_synthetic, get_true_ordering){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_ORDERING") == 0) return "ordered";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_TRUE(config.ordering());
}

TEST(configuration_synthetic, get_false_ordering){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_ORDERING") == 0) return "unordered";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_FALSE(config.ordering());
}

TEST(configuration_synthetic, get_wrong_ordering){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_ORDERING") == 0) return "grppi";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_TRUE(config.ordering());
}

TEST(configuration_synthetic, get_good_queue_size){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_QUEUE_SIZE") == 0) return "20";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(20, config.queue_size());
}

TEST(configuration_synthetic, get_zero_queue_size){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_QUEUE_SIZE") == 0) return "0";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(config.default_queue_size, config.queue_size());
}

TEST(configuration_synthetic, get_negative_queue_size){
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_QUEUE_SIZE") == 0) return "-20";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(config.default_queue_size, config.queue_size());
}

TEST(configuration_synthetic, get_blocking_mode) {
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_QUEUE_MODE") == 0) return "blocking";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(queue_mode::blocking, config.mode());
}

TEST(configuration_synthetic, get_lockfree_mode) {
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_QUEUE_MODE") == 0) return "lockfree";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(queue_mode::lockfree, config.mode());
}

TEST(configuration_synthetic, get_unknown_mode) {
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_QUEUE_MODE") == 0) return "unknown";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_EQ(queue_mode::blocking, config.mode());
}

TEST(configuration_synthetic, get_seq_backend) {
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_DYN_BACKEND") == 0) return "seq";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_STREQ("seq", config.dynamic_backend().c_str());
}

TEST(configuration_synthetic, get_omp_backend) {
  struct getter {
    const char * operator()(char const * var_name) {
      if (strcmp(var_name,"GRPPI_DYN_BACKEND") == 0) return "omp";
      return nullptr;
    }
  };

  configuration<getter> config;
  EXPECT_STREQ("omp", config.dynamic_backend().c_str());
}

