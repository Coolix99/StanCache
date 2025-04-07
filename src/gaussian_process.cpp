#include "stancache/gaussian_process.hpp"
#include "stancache/kernel.hpp"
#include "stancache/kd_adaptor.hpp"
#include <nanoflann.hpp>


#include <Eigen/Cholesky>

namespace stancache {

void GaussianProcess::add_sample(const Eigen::VectorXd& x, double y_val) {
    X.push_back(x);
    if (y.size() == 0) y = Eigen::VectorXd::Constant(1, y_val);
    else {
        Eigen::VectorXd y_new(y.size() + 1);
        y_new << y, y_val;
        y = y_new;
    }
}

std::pair<double, double> GaussianProcess::predict(const Eigen::VectorXd& x_star) {
    int n = X.size();
    if (n == 0) return {0.0, 1e6};

    Eigen::MatrixXd K(n, n);
    Eigen::VectorXd k_star(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double kij = squared_exponential_kernel(X[i], X[j], sigma_f, length_scale);
            K(i, j) = kij;
            K(j, i) = kij;
        }
        k_star(i) = squared_exponential_kernel(X[i], x_star, sigma_f, length_scale);
    }

    K += noise * Eigen::MatrixXd::Identity(n, n);
    Eigen::LLT<Eigen::MatrixXd> llt(K);
    Eigen::VectorXd alpha = llt.solve(y);

    double mean = k_star.dot(alpha);

    Eigen::VectorXd v = llt.matrixL().solve(k_star);
    double var = squared_exponential_kernel(x_star, x_star, sigma_f, length_scale) - v.dot(v);
    return {mean, std::sqrt(std::max(var, 1e-10))};
}

std::pair<double, double> GaussianProcess::predict_local(const Eigen::VectorXd& x_star, int k) const {
    int N = X.size();
    if (N == 0) return {0.0, 1e6};
    k = std::min(k, N);

    using namespace nanoflann;
    using Adaptor = stancache::PointCloudAdaptor;
    using KDTree = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<double, Adaptor>,
        Adaptor,
        -1,  // dynamic dimensionality
        size_t>;

    Adaptor adaptor(X, x_star.size());
    KDTree index(x_star.size(), adaptor, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    std::vector<size_t> ret_indexes(k);
    std::vector<double> out_dists(k);
    KNNResultSet<double> resultSet(k);
    resultSet.init(ret_indexes.data(), out_dists.data());
    index.findNeighbors(resultSet, x_star.data(), nanoflann::SearchParameters(10, 0.0));


    // Collect local neighbors
    Eigen::MatrixXd Xk(k, x_star.size());
    Eigen::VectorXd yk(k);
    for (int i = 0; i < k; ++i) {
        int idx = ret_indexes[i];
        Xk.row(i) = X[idx];
        yk(i) = y(idx);
    }

    // Build local kernel matrix
    Eigen::MatrixXd K(k, k);
    Eigen::VectorXd k_star(k);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            double kij = squared_exponential_kernel(Xk.row(i), Xk.row(j), sigma_f, length_scale);
            K(i, j) = kij;
            K(j, i) = kij;
        }
        k_star(i) = squared_exponential_kernel(Xk.row(i), x_star, sigma_f, length_scale);
    }

    K += noise * Eigen::MatrixXd::Identity(k, k);
    Eigen::LLT<Eigen::MatrixXd> llt(K);
    Eigen::VectorXd alpha = llt.solve(yk);
    double mean = k_star.dot(alpha);

    Eigen::VectorXd v = llt.matrixL().solve(k_star);
    double var = squared_exponential_kernel(x_star, x_star, sigma_f, length_scale) - v.dot(v);

    return {mean, std::sqrt(std::max(var, 1e-10))};
}

}  // namespace stancache

