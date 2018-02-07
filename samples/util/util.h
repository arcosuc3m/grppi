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

#ifndef GRPPI_SAMPLES_UTIL_H
#define GRPPI_SAMPLES_UTIL_H

#include "dyn/dynamic_execution.h"

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
