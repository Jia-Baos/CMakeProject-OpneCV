#pragma once
#ifndef PROJECTION_H
#define PROJECTION_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void GaussSmoothOriHist(int* hist, int n);

cv::Mat GetVerProjImage(const cv::Mat& image);

cv::Mat GetHorProjImage(const cv::Mat& image);

bool GetVerProjRegions(const cv::Mat& image, std::vector<int>& regionIndex);

bool GetHorProjRegions(const cv::Mat& image, std::vector<int>& regionIndex);

#endif // !PROJECTION_H
