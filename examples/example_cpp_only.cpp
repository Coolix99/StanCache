#include "stancache/interpolator.hpp"
#include <iostream>
#include <cmath>

int main() {
    using namespace stancache;

    int expensive_call_count = 0;
    auto expensive = [&](const Eigen::VectorXd& x) {
        ++expensive_call_count;
        return std::exp(-x[0]*x[0] - x[1]*x[1]);  // Gaussian bump
    };

    GPInterpolatorWithCache f(expensive, 0.01);  // Use cached GP if error < 0.01

    for (double x = -2.0; x <= 2.0; x += 0.2) {
        for (double y = -2.0; y <= 2.0; y += 0.2) {
            Eigen::Vector2d v(x, y);
            double val = f(v);
            std::cout << "f(" << v.transpose() << ") = " << val << "\n";
        }
    }

    std::cout << "Total expensive calls: " << expensive_call_count << "\n";
    return 0;
}