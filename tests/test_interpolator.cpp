#include "stancache/interpolator.hpp"
#include <iostream>
#include <cmath>

double expensive(const Eigen::VectorXd& x) {
    return std::sin(x[0]) + std::cos(x[1]);
}

int main() {
    stancache::GPInterpolatorWithCache f(expensive, 0.05);

    for (double x = 0.0; x < 3.0; x += 0.5) {
        Eigen::Vector2d v(x, x * 0.3);
        double y = f(v);
        std::cout << "f(" << v.transpose() << ") = " << y << std::endl;
    }

    // Call again to test interpolation
    Eigen::Vector2d v(1.0, 0.3);
    double y = f(v);
    std::cout << "Second call f(" << v.transpose() << ") = " << y << std::endl;

    return 0;
}
