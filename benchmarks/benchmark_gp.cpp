#include "stancache/interpolator.hpp"
#include "stancache/benchmark.hpp"
#include <iostream>
#include <cmath>

int main() {
    using namespace stancache;

    int expensive_call_count = 0;

    auto expensive = [&](const Eigen::VectorXd& x) {
        ++expensive_call_count;
        return std::sin(x[0]) + std::cos(x[1]);
    };

    GPInterpolatorWithCache f(expensive, 0.05);
    Benchmark benchmark;

    for (double x = 0.0; x <= 5.0; x += 0.1) {
        for (double y = 0.0; y <= 5.0; y += 0.1) {
            Eigen::Vector2d v(x, y);
            double true_val = std::sin(x) + std::cos(y);
            double pred_val = f(v);
            benchmark.add(pred_val, true_val);
        }
    }

    benchmark.report();
    std::cout << "Total expensive function calls: " << expensive_call_count << "\n";
    return 0;
}