#ifndef GRPPI_COMMON_FARM_PATTERN_H
#define GRPPI_COMMON_FARM_PATTERN_H

#include <type_traits>

namespace grppi {

class sequential_execution;

template <typename E, typename Transformer>
class farm_t {
public:
  farm_t(const E & e, Transformer && t) :
    exec_{e}, transformer_{t}
  {}

  template <typename I>
  auto operator()(I && item) {
    return transformer_(std::forward<I>(item));
  }

private:
  const E & exec_;
  Transformer transformer_;
};

}

#endif
