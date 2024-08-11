#ifndef EPICAUX_HPP
#define EPICAUX_HPP

#include <string>
#include <vector>
#include <array>
#include <queue>
#include <unordered_map>
#include <opencv2/opencv.hpp>

/* structure for distance transform parameters */
struct dt_params_t
{
    int max_iter = 40;
    float min_change = 1.0f;
};

struct csr_matrix
{
    std::vector<int> indices; // indices of columns
    std::vector<int> indptr;  // indices of indices of rows
    std::vector<float> data;  // row i contains values data[indptr[i]:indptr[i+1]] at columns indices[indptr[i]:indptr[i+1]]
    int nr;
    int nc;
};

template<typename T>
inline T MY_MIN(T a, T b) {
    return a < b ? a : b;
}

template<typename T>
inline T MY_MAX(T a, T b) {
    return a > b ? a : b;
}

template<typename T>
inline void MY_SWAP(T& a, T& b) {
    std::swap(a, b);
}

template<typename T>
inline bool MY_BETWEEN(T min, T  val, T  max) {
    return min <= val && val <= max;
}

/// @brief Compute the closest seeds using a geodesic distance for a subset of points, and the assignment of each pixel to the closest seeds
/// @param best: output containing the closest seeds for each query point
/// @param dist: output containing the distances to the closest seeds
/// @param labels: output containing the assignment of each pixel to the closest seed
/// @param seeds: 2D positions of the seeds
/// @param cost: cost of going throw a pixel (ie that defines the geodesic distance)
/// @param dt_params: distance transform parameters (NULL for default parameters)
/// @param  pixels: 2D positions of the query points
/// @return None
void dist_trf_nnfield_subset(cv::Mat& best, cv::Mat& dist, cv::Mat& labels,
    const std::vector<cv::Vec4f>& seeds,
    const cv::Mat& cost, dt_params_t& dt_params,
    const std::vector<cv::Vec4f>& pixels, const int n_thread);

#endif // ! EPICAUX_HPP
