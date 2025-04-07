#ifndef STANCACHE_KERNEL_HPP
#define STANCACHE_KERNEL_HPP

#include <Eigen/Dense>

namespace stancache {

double squared_exponential_kernel(const Eigen::VectorXd& x1,
                                   const Eigen::VectorXd& x2,
                                   double sigma_f = 1.0,
                                   double length_scale = 1.0);

}  // namespace stancache

#endif