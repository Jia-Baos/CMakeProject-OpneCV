#pragma once
#ifndef MYUTILS_H
#define MYUTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>

bool Get_Gauss_Pyramid_Down(const cv::Mat src, std::vector<cv::Mat>& Gauss_Pyramid, int layers = 3)
{
	// 构建图像金字塔，共四层，可直接在此处修改金字塔的层数
	cv::Mat current_img = src.clone();
	Gauss_Pyramid.push_back(current_img);
	for (int i = 0; i < layers - 1; i++)
	{
		cv::Mat temp_img;
		cv::pyrDown(current_img, temp_img, cv::Size(current_img.cols / 2, current_img.rows / 2));
		Gauss_Pyramid.push_back(temp_img);
		current_img = temp_img;
	}
	return 0;
}

bool Get_Gauss_Pyramid_Up(const cv::Mat src, std::vector<cv::Mat>& Gauss_Pyramid, int layers = 3)
{
	// 构建图像金字塔，共四层，可直接在此处修改金字塔的层数
	cv::Mat current_img = src.clone();
	Gauss_Pyramid.push_back(current_img);
	for (int i = 0; i < layers - 1; i++)
	{
		cv::Mat temp_img;
		cv::pyrUp(current_img, temp_img, cv::Size(current_img.cols * 2, current_img.rows * 2));
		Gauss_Pyramid.push_back(temp_img);
		current_img = temp_img;
	}
	return 0;
}

#endif // !MYUTILS_H
