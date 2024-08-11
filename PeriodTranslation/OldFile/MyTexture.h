#pragma once
#ifndef MYTEXTURE_H
#define MYTEXTURE_H

#include <iostream>
#include <opencv2/opencv.hpp>

bool Capture_Texture(const cv::Mat src, cv::Mat& dst)
{
	cv::Mat src_copy = src.clone();
	cv::Mat dst_copy = cv::Mat::zeros(src_copy.size(), CV_8UC1);

	uchar* srcData = src_copy.data;
	uchar* dstData = dst_copy.data;

	for (int j = 0; j < src_copy.cols; j++)
	{
		for (int i = 0; i < src_copy.rows; i++)
		{
			int b = *(srcData + src_copy.step * i + src_copy.channels() * j + 0);
			int g = *(srcData + src_copy.step * i + src_copy.channels() * j + 1);
			int r = *(srcData + src_copy.step * i + src_copy.channels() * j + 2);

			int gray = 0.299 * r + 0.587 * g + 0.114 * b;

			if (10 < (r - g))
			{
				*(dstData + dst_copy.step * i + dst_copy.channels() * j + 0) = r;
			}
			else
			{
				*(dstData + dst_copy.step * i + dst_copy.channels() * j + 0) = 205;
			}
		}
	}
	// cv::normalize(dst_copy, dst_copy, 0, 255, cv::NORM_MINMAX);
	dst = dst_copy.clone();
	return 0;
}

#endif // !MYTEXTURE_H
