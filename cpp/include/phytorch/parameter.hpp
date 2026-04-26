#pragma once

#include <limits>
#include <string_view>

namespace phytorch {

// Mirror of phytorch.models.base.Model.parameter_info() entries.
// Models declare a `static constexpr std::array<ParameterInfo, N> info{...}`
// at compile time so the optimizer can read defaults / bounds / labels
// without virtual dispatch.
struct ParameterInfo {
    std::string_view name;
    std::string_view symbol;
    std::string_view units;
    std::string_view description;
    double           default_value;
    double           lower_bound;
    double           upper_bound;
};

inline constexpr double kNoLowerBound = -std::numeric_limits<double>::infinity();
inline constexpr double kNoUpperBound =  std::numeric_limits<double>::infinity();

}  // namespace phytorch
