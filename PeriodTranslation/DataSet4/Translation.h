#pragma once
#ifndef TRANSLATION_H
#define TRANSLATION_H

#include <iostream>
#include <opencv2/opencv.hpp>

bool ComputeParameter(const cv::Mat& image, const cv::Mat& template1,
	std::vector<double>& similarity, std::vector<cv::Point>& location);

cv::Mat PeriodTranslation_Mode(const cv::Mat& image, const cv::Mat& template1);

cv::Mat EdgeProcess(const cv::Mat& image);

cv::Mat PeriodTranslation_BGR(const cv::Mat& src, const int position = 100);

cv::Mat PeriodTranslation_GRAY(const cv::Mat& src, const int position = 200);

#endif // !TRANSLATION_H
