// ========== src/kernel.cpp ==========
#include "stancache/kernel.hpp"

namespace stancache {

double squared_exponential_kernel(const Eigen::VectorXd& x1,
                                   const Eigen::VectorXd& x2,
                                   double sigma_f,
                                   double length_scale) {
    return sigma_f * sigma_f * std::exp(-(x1 - x2).squaredNorm() / (2.0 * length_scale * length_scale));
}

}  // namespace stancache