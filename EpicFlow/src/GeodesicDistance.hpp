#ifndef GEODESICDISTANCE_HPP
#define GEODESICDISTANCE_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

/// @brief Distance Type
enum DistType
{
	euclidean,
	intensity,
	geodesic
};

/// @brief Examine the seed's distance
/// @param dist Distance map
/// @param seeds Seeds
void CheckSeeds(const cv::Mat& dist, const std::vector<cv::Point2f> seeds);

/// @brief Get the forward and backward kernel and correspond spatial distance
/// @param kernel Forward or backward kernel
/// @param squared_dist Correspond spatial distance
/// @param backward Forward or backward
void GetKernel(std::unordered_map<std::string, std::vector<int>>& kernel,
	std::vector<float>& squared_dist, bool backward);

/// @brief Get the distance between two pixels
/// @param alpha Scaling factor of distance
/// @param p_val Value of p
/// @param q_val Value of q
/// @param scaling_factor Scaling factor between spatial and intensity distance
/// @param squared_dist spatial distance
/// @param dist_type Distance type
/// @return Distance between two pixels
float GetDistance(const float alpha, const float p_val, const float q_val,
	const float scaling_factor, const float squared_dist,
	const DistType dist_type);

/// @brief Pass image forward or backward to compute distance
/// @param edges Cost map Computed using SED
/// @param dist Distance map
/// @param kernel Forward or backward kernel
/// @param squared_dist Squared_dist Correspond spatial distance
/// @param alpha Scaling factor of distance
/// @param scaling_factor Scaling factor between spatial and intensity distance
/// @param dist_type Distance type
/// @param backward Forward or backward
void Pass2D(const cv::Mat& edges, cv::Mat& dist,
	std::unordered_map<std::string, std::vector<int>>& kernel,
	std::vector<float>& squared_dist, const float alpha,
	const float scaling_factor,
	const DistType dist_type,
	bool backward);

void GetDMRasterscan(const cv::Mat& edges, cv::Mat& dist,
	const std::vector<cv::Point2f>& seeds, const int its,
	const float scaling_factor,
	const DistType dist_type);

#endif // !GEODESICDISTANCE_HPP
