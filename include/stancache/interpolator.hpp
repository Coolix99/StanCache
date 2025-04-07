#ifndef STANCACHE_INTERPOLATOR_HPP
#define STANCACHE_INTERPOLATOR_HPP

#include "gaussian_process.hpp"
#include <functional>

namespace stancache {

class GPInterpolatorWithCache {
public:
    using ExpensiveFn = std::function<double(const Eigen::VectorXd&)>;

    GPInterpolatorWithCache(ExpensiveFn f, double error_threshold = 0.1);

    double operator()(const Eigen::VectorXd& x);

private:
    GaussianProcess gp;
    ExpensiveFn expensive_function;
    double error_threshold;
};

}  // namespace stancache

#endif
