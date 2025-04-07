#include "stancache/interpolator.hpp"

namespace stancache {

GPInterpolatorWithCache::GPInterpolatorWithCache(ExpensiveFn f, double error_threshold)
    : expensive_function(f), error_threshold(error_threshold) {}

double GPInterpolatorWithCache::operator()(const Eigen::VectorXd& x) {
    int k = 30;  
    auto [mean, std] = gp.predict_local(x, k);
    if (std < error_threshold)
        return mean;

    double exact = expensive_function(x);
    gp.add_sample(x, exact);
    return exact;
}

}
