#pragma once
#ifndef PROJECTION_H
#define PROJECTION_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void GaussSmoothOriHist(int* hist, int n);

cv::Mat GetVerProjImage(const cv::Mat& image);

int GetVerProjRegions(const cv::Mat& image);

cv::Mat GetHorProjImage(const cv::Mat& image);

#endif // !PROJECTION_H
