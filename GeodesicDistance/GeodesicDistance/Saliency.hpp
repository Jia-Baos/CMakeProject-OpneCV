#ifndef SALIENCY_HPP
#define SALIENCY_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#define USE_OPENMP
#define num_workers 4

/// @brief Compute the Gradient of gray of RGB image
/// @param src Input image
/// @param Fx Horizontal gradient
/// @param Fy Vertical gradient
void ComputeGradient(const cv::Mat& src, cv::Mat& Fx, cv::Mat& Fy);

/// @brief Compute the Saliency Map
/// @param src Input image
/// @param dst Salinecy Map
/// @param sigma_image Smoothing factor
/// @param sigma_matrix Integrate autocorrelation matrix fatcor
void Saliency(const cv::Mat& src, cv::Mat& dst, float sigma_image, float sigma_matrix);

#endif // !SALIENCY_HPP
