#ifndef STANCACHE_GAUSSIAN_PROCESS_HPP
#define STANCACHE_GAUSSIAN_PROCESS_HPP

#include <Eigen/Dense>
#include <vector>
#include <utility>

namespace stancache {

class GaussianProcess {
public:
    void add_sample(const Eigen::VectorXd& x, double y);
    std::pair<double, double> predict(const Eigen::VectorXd& x_star);
    std::pair<double, double> predict_local(const Eigen::VectorXd& x_star, int k) const;


private:
    std::vector<Eigen::VectorXd> X;
    Eigen::VectorXd y;
    double sigma_f = 1.0;
    double length_scale = 1.0;
    double noise = 1e-6;
};

}  // namespace stancache

#endif