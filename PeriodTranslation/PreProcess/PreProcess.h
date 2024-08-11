#pragma once
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat MainProcess(const cv::Mat& image);

cv::Mat BilateralFilter(const cv::Mat& image, int width, int sigmaSpace, int sigmaColor);

#endif // !PREPROCESS_H
