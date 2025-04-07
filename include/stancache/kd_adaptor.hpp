#ifndef STANCACHE_KD_ADAPTOR_HPP
#define STANCACHE_KD_ADAPTOR_HPP

#include <Eigen/Dense>
#include <vector>
#include <nanoflann.hpp>

namespace stancache {

struct PointCloudAdaptor {
    const std::vector<Eigen::VectorXd>& pts;
    const int dim;

    PointCloudAdaptor(const std::vector<Eigen::VectorXd>& points, int dimension)
        : pts(points), dim(dimension) {}

    // Required by nanoflann
    inline size_t kdtree_get_point_count() const {
        return pts.size();
    }

    inline double kdtree_get_pt(const size_t idx, const size_t d) const {
        return pts[idx][d];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

}  // namespace stancache

#endif
