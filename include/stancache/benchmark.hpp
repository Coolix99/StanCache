#ifndef STANCACHE_BENCHMARK_HPP
#define STANCACHE_BENCHMARK_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace stancache {

class Benchmark {
public:
    void add(double predicted, double actual) {
        errors.push_back(std::abs(predicted - actual));
    }

    void report() const {
        if (errors.empty()) return;
        double avg = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        double min = *std::min_element(errors.begin(), errors.end());
        double max = *std::max_element(errors.begin(), errors.end());

        std::cout << "Average absolute error: " << avg << "\n";
        std::cout << "Min error: " << min << ", Max error: " << max << "\n";
        std::cout << "Sample count: " << errors.size() << "\n";
    }

private:
    std::vector<double> errors;
};

} // namespace stancache

#endif
