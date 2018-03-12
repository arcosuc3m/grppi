/**
* @version		GrPPI v0.3
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
#include <experimental/optional>
#include <numeric>

#include <gtest/gtest.h>

#include "grppi.h"
#include "dyn/dynamic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class context_test : public ::testing::Test {
public:
  T execution_;
  dynamic_execution dyn_execution_{execution_};
  sequential_execution seq_;
  parallel_execution_native thr_;
#ifdef GRPPI_OMP
  parallel_execution_omp omp_;
#endif
#ifdef GRPPI_TBB
  parallel_execution_tbb tbb_;
#endif
#ifdef GRPPI_FF
  parallel_execution_ff ff_;
#endif
  // Variables
  int out;
  int counter;

  // Vectors
  vector<int> v{};
  vector<vector<int> > v2{};

  // Invocation counter
  std::atomic<int> invocations_init{0};
  std::atomic<int> invocations_last{0};
  std::atomic<int> invocations_intermediate{0};
  std::atomic<int> invocations_intermediate2{0};
  std::atomic<int> invocations_intermediate3{0};

  void setup_three_stages() {
    counter = 5;
    out = 0;
  }

  template <typename E>
  void run_three_stages_farm_with_sequential(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {}; 
      },
      grppi::run_with(this->seq_,
        grppi::farm(2,
          [this](int x) {
            invocations_intermediate++;
            return x*2;
          }
        )
      ),
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  template <typename E>
  void run_three_stages_farm_with_native(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {}; 
      },
      grppi::run_with(this->thr_,
        grppi::farm(2,
          [this](int x) {
            invocations_intermediate++;
            return x*2;
          }
        )
      ),
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  template <typename E>
  void run_three_stages_farm_with_omp(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {}; 
      },
#ifdef GRPPI_OMP
      grppi::run_with(this->omp_,
        grppi::farm(2,
          [this](int x) {
            invocations_intermediate++;
            return x*2;
          }
        )
      ),
#endif
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  template <typename E>
  void run_three_stages_farm_with_tbb(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {}; 
      },
#ifdef GRPPI_TBB
      grppi::run_with(this->tbb_,
        grppi::farm(2,
          [this](int x) {
            invocations_intermediate++;
            return x*2;
          }
        )
      ),
#endif
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }

  template <typename E>
  void run_three_stages_farm_with_ff(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_FF
      grppi::run_with(this->ff_,
        grppi::farm(2,
          [this](int x) {
            invocations_intermediate++;
            return x*2;
          }
        )
      ),
#endif
      [this](int x) {
        invocations_last++;
        out += x;
      });
  }



  void check_three_stages() {
    EXPECT_EQ(6, invocations_init); 
    EXPECT_EQ(5, invocations_last); 
    EXPECT_EQ(5, invocations_intermediate);
    EXPECT_EQ(0, invocations_intermediate2);
    EXPECT_EQ(30, out);
  }

  void setup_composed() {
    counter = 5;
    out = 0;
  }

  template <typename E>
  void run_composed_pipeline_with_sequential(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      grppi::run_with(this->seq_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          [this](int x) {
            invocations_intermediate2++;
            x *= 2;
            return x;
          }
        )
      ),
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_composed_pipeline_with_native(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      grppi::run_with(this->thr_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          [this](int x) {
            invocations_intermediate2++;
            x *= 2;
            return x;
          }
        )
      ),
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_composed_pipeline_with_omp(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_OMP
      grppi::run_with(this->omp_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          [this](int x) {
            invocations_intermediate2++;
            x *= 2;
            return x;
          }
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_composed_pipeline_with_tbb(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_TBB
      grppi::run_with(this->tbb_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          [this](int x) {
            invocations_intermediate2++;
            x *= 2;
            return x;
          }
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_composed_pipeline_with_ff(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_FF
      grppi::run_with(this->ff_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          [this](int x) {
            invocations_intermediate2++;
            x *= 2;
            return x;
          }
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }


  void check_composed() {
    EXPECT_EQ(6, invocations_init); 
    EXPECT_EQ(5, invocations_last); 
    EXPECT_EQ(5, invocations_intermediate);
    EXPECT_EQ(5, invocations_intermediate2);
    EXPECT_EQ(110, out);
  }

  void setup_double_composed() {
    counter = 5;
    out = 0;
  }

  template <typename E>
  void run_double_composed_pipeline_sequential_sequential(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      grppi::run_with(this->seq_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          grppi::run_with(this->seq_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_double_composed_pipeline_sequential_native(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      grppi::run_with(this->seq_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          grppi::run_with(this->thr_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_double_composed_pipeline_sequential_omp(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_OMP
      grppi::run_with(this->seq_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
        grppi::run_with(this->omp_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_double_composed_pipeline_sequential_tbb(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_TBB
      grppi::run_with(this->seq_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
        grppi::run_with(this->tbb_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_double_composed_pipeline_sequential_ff(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_FF
      grppi::run_with(this->seq_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
        grppi::run_with(this->ff_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }



  template <typename E>
  void run_double_composed_pipeline_native_sequential(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
      grppi::run_with(this->thr_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
          grppi::run_with(this->seq_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_double_composed_pipeline_omp_sequential(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_OMP
      grppi::run_with(this->omp_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
        grppi::run_with(this->seq_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_double_composed_pipeline_tbb_sequential(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_TBB
      grppi::run_with(this->tbb_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
        grppi::run_with(this->seq_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }

  template <typename E>
  void run_double_composed_pipeline_ff_sequential(const E & e) {
    grppi::pipeline(e,
      [this,i=0,max=counter]() mutable -> optional<int> {
        invocations_init++;
        if (++i<=max) return i;
        else return {};
      },
#ifdef GRPPI_FF
      grppi::run_with(this->ff_,
        grppi::pipeline(
          [this](int x) {
            invocations_intermediate++;
            return x*x;
          },
        grppi::run_with(this->seq_,
            grppi::pipeline(
              [this](int x) {
                invocations_intermediate2++;
                return x+1;
              },
              [this](int x) {
                invocations_intermediate3++;
                x *= 2;
                return x;
              }
            )
          )
        )
      ),
#endif
      [this](int x) {
          invocations_last++;
          out +=x;
      });
  }



  void check_double_composed() {
    EXPECT_EQ(6, invocations_init); 
    EXPECT_EQ(5, invocations_last); 
    EXPECT_EQ(5, invocations_intermediate);
    EXPECT_EQ(5, invocations_intermediate2);
    EXPECT_EQ(5, invocations_intermediate3);
    EXPECT_EQ(120, out);
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(context_test, executions);

TYPED_TEST(context_test, static_three_stages_farm_seq)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_sequential(this->execution_);
  this->check_three_stages();
}

TYPED_TEST(context_test, static_three_stages_farm_nat)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_native(this->execution_);
  this->check_three_stages();
}

#ifdef GRPPI_OMP
TYPED_TEST(context_test, static_three_stages_farm_omp)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_omp(this->execution_);
  this->check_three_stages();
}
#endif

#ifdef GRPPI_TBB
TYPED_TEST(context_test, static_three_stages_farm_tbb)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_tbb(this->execution_);
  this->check_three_stages();
}
#endif

#ifdef GRPPI_FF
TYPED_TEST(context_test, static_three_stages_farm_ff)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_ff(this->execution_);
  this->check_three_stages();
}
#endif


TYPED_TEST(context_test, dyn_three_stages_farm_seq)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_sequential(this->dyn_execution_);
  this->check_three_stages();
}

TYPED_TEST(context_test, dyn_three_stages_farm_nat)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_native(this->dyn_execution_);
  this->check_three_stages();
}

#ifdef GRPPI_OMP
TYPED_TEST(context_test, dyn_three_stages_farm_omp)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_omp(this->dyn_execution_);
  this->check_three_stages();
}
#endif

#ifdef GRPPI_TBB
TYPED_TEST(context_test, dyn_three_stages_farm_tbb)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_tbb(this->dyn_execution_);
  this->check_three_stages();
}
#endif

#ifdef GRPPI_FF
TYPED_TEST(context_test, dyn_three_stages_farm_ff)
{
  this->setup_three_stages();
  this->run_three_stages_farm_with_ff(this->dyn_execution_);
  this->check_three_stages();
}
#endif

TYPED_TEST(context_test, static_composed_pipeline_seq)
{
  this->setup_composed();
  this->run_composed_pipeline_with_sequential(this->execution_);
  this->check_composed();
}

TYPED_TEST(context_test, static_composed_pipeline_nat)
{
  this->setup_composed();
  this->run_composed_pipeline_with_native(this->execution_);
  this->check_composed();
}

#ifdef GRPPI_OMP
TYPED_TEST(context_test, static_composed_pipeline_omp)
{
  this->setup_composed();
  this->run_composed_pipeline_with_omp(this->execution_);
  this->check_composed();
}
#endif

#ifdef GRPPI_TBB
TYPED_TEST(context_test, static_composed_pipeline_tbb)
{
  this->setup_composed();
  this->run_composed_pipeline_with_tbb(this->execution_);
  this->check_composed();
}
#endif

#ifdef GRPPI_FF
TYPED_TEST(context_test, static_composed_pipeline_ff)
{
  this->setup_composed();
  this->run_composed_pipeline_with_ff(this->execution_);
  this->check_composed();
}
#endif

TYPED_TEST(context_test, dyn_composed_pipeline_seq)
{
  this->setup_composed();
  this->run_composed_pipeline_with_sequential(this->dyn_execution_);
  this->check_composed();
}

TYPED_TEST(context_test, dyn_composed_pipeline_nat)
{
  this->setup_composed();
  this->run_composed_pipeline_with_native(this->dyn_execution_);
  this->check_composed();
}

#ifdef GRPPI_OMP
TYPED_TEST(context_test, dyn_composed_pipeline_omp)
{
  this->setup_composed();
  this->run_composed_pipeline_with_omp(this->dyn_execution_);
  this->check_composed();
}
#endif

#ifdef GRPPI_TBB
TYPED_TEST(context_test, dyn_composed_pipeline_tbb)
{
  this->setup_composed();
  this->run_composed_pipeline_with_tbb(this->dyn_execution_);
  this->check_composed();
}
#endif

#ifdef GRPPI_FF
TYPED_TEST(context_test, dyn_composed_pipeline_ff)
{
  this->setup_composed();
  this->run_composed_pipeline_with_ff(this->dyn_execution_);
  this->check_composed();
}
#endif

TYPED_TEST(context_test, static_double_composed_pipeline_seq_seq)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_sequential_sequential(this->execution_);
  this->check_double_composed();
}

TYPED_TEST(context_test, static_double_composed_pipeline_seq_thr)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_sequential_native(this->execution_);
  this->check_double_composed();
}

#ifdef GRPPI_OMP
TYPED_TEST(context_test, static_double_composed_pipeline_seq_omp)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_sequential_omp(this->execution_);
  this->check_double_composed();
}
#endif

#ifdef GRPPI_TBB
TYPED_TEST(context_test, static_double_composed_pipeline_seq_tbb)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_sequential_tbb(this->execution_);
  this->check_double_composed();
}
#endif

#ifdef GRPPI_FF
TYPED_TEST(context_test, static_double_composed_pipeline_seq_FF)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_sequential_ff(this->execution_);
  this->check_double_composed();
}
#endif

TYPED_TEST(context_test, static_double_composed_pipeline_thr_seq)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_native_sequential(this->execution_);
  this->check_double_composed();
}

#ifdef GRPPI_OMP
TYPED_TEST(context_test, static_double_composed_pipeline_omp_seq)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_omp_sequential(this->execution_);
  this->check_double_composed();
}
#endif

#ifdef GRPPI_TBB
TYPED_TEST(context_test, static_double_composed_pipeline_tbb_seq)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_tbb_sequential(this->execution_);
  this->check_double_composed();
}
#endif

#ifdef GRPPI_FF
TYPED_TEST(context_test, static_double_composed_pipeline_FF_seq)
{
  this->setup_double_composed();
  this->run_double_composed_pipeline_ff_sequential(this->execution_);
  this->check_double_composed();
}
#endif


