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
#ifndef GRPPI_SAMPLES_UTIL_H
#define GRPPI_SAMPLES_UTIL_H

#include "grppi/dyn/dynamic_execution.h"

grppi::dynamic_execution execution_mode(const std::string & opt) {
  using namespace grppi;
  if ("seq" == opt) return sequential_execution{};
  if ("thr" == opt) return parallel_execution_native{};
  if ("omp" == opt) return parallel_execution_omp{};
  if ("tbb" == opt) return parallel_execution_tbb{};
  if ("ff" == opt)  return parallel_execution_ff{};
  return {};
}

template <typename F, typename ...Args>
bool run_test(const std::string & mode, F && f, Args && ... args) {
  auto e = execution_mode(mode);
  if (e.has_execution()) {
    f(e, std::forward<Args>(args)...);
    return true;
  }
  return false;
}

void print_available_modes(std::ostream & os) {
  using namespace std;
  using namespace grppi;

  if (is_supported<sequential_execution>()) {
    os << "    seq -> Sequential execution" << endl;
  }

  if (is_supported<parallel_execution_native>()) {
    os << "    thr -> ISO Threads backend" << endl;
  }

  if (is_supported<parallel_execution_tbb>()) {
    os << "    tbb -> Intel TBB backend" << endl;
  }

  if (is_supported<parallel_execution_omp>()) {
    os << "    omp -> OpenMP backend" << endl;
  }

  if (is_supported<parallel_execution_ff>()) {
    os << "    ff -> FastFlow backend" << endl;
  }
}

#endif
