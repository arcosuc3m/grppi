#ifndef GRPPI_OPTIONAL_H
#define GRPPI_OPTIONAL_H

#if __cplusplus < 201703L
#include <experimental/optional>
#else
#include <optional>
#endif

namespace grppi {

#if __cplusplus < 201703L
  template <typename T>
  using optional = std::experimental::optional<T>;
#else
  template <typename T>
  using optional = std::optional<T>;
#endif
}

#endif //GRPPI_OPTIONAL_H
