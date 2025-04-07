#include "stancache/gaussian_process.hpp"
#include <iostream>
#include <cassert>

int main() {
    stancache::GaussianProcess gp;
    gp.add_sample(Eigen::Vector2d(0.0, 0.0), 1.0);
    gp.add_sample(Eigen::Vector2d(1.0, 1.0), 2.0);

    auto [mean, std] = gp.predict(Eigen::Vector2d(0.5, 0.5));
    std::cout << "Prediction: mean = " << mean << ", std = " << std << std::endl;

    assert(std >= 0.0);
    return 0;
}