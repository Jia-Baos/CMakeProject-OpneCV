#pragma once
#ifndef MSER_H
#define MSER_H

#include <iostream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

cv::Mat MSERDetection(const cv::Mat& image);

#endif // !MSER_H
