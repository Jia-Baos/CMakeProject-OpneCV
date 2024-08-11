#pragma once
#ifndef TRANSLATION_H
#define TRANSLATION_H

#include <iostream>
#include <opencv2/opencv.hpp>

void ComputeParameter(const cv::Mat& image, const cv::Mat& template1, const cv::Mat& template2,
	std::vector<double>& similarity, std::vector<cv::Point>& location);

void EdgeProcess(const cv::Mat& src, cv::Mat& dst);

void PeriodTranslation_BGR(const cv::Mat& src, cv::Mat& dst, const int distance = 100);

cv::Mat PeriodTranslation_Mode(const cv::Mat& image, const cv::Mat& template1, const cv::Mat& template2);

#endif // !TRANSLATION_H
