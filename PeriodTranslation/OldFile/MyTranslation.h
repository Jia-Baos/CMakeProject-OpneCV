#pragma once
#ifndef MYTRANSLATION_H
#define MYTRANSLATION_H

#include <iostream>
#include <opencv2/opencv.hpp>

bool Period_Translation_BGR(const cv::Mat src, cv::Mat& dst, const int position=100)
{
	cv::Mat src_copy = src.clone();
	cv::Mat dst_copy = cv::Mat(src.size(), src.type());

	// 创建指针指向src_copy的首地址
	uchar* srcData = src_copy.data;
	uchar* dstData = dst_copy.data;

	for (int j = 0; j < src_copy.cols; j++)
	{
		for (int i = 0; i < src_copy.rows; i++)
		{
			int b = *(srcData + src_copy.step * i + src_copy.channels() * j + 0);
			int g = *(srcData + src_copy.step * i + src_copy.channels() * j + 1);
			int r = *(srcData + src_copy.step * i + src_copy.channels() * j + 2);

			if (j < position)
			{
				*(dstData + dst_copy.step * i + dst_copy.channels() * (src_copy.cols - position + j) + 0) = b;
				*(dstData + dst_copy.step * i + dst_copy.channels() * (src_copy.cols - position + j) + 1) = g;
				*(dstData + dst_copy.step * i + dst_copy.channels() * (src_copy.cols - position + j) + 2) = r;
			}
			else
			{
				*(dstData + dst_copy.step * i + dst_copy.channels() * (j - position) + 0) = b;
				*(dstData + dst_copy.step * i + dst_copy.channels() * (j - position) + 1) = g;
				*(dstData + dst_copy.step * i + dst_copy.channels() * (j - position) + 2) = r;
			}
		}
	}

	dst = dst_copy.clone();
	return 0;
}


bool Period_Translation_GRAY(const cv::Mat src, cv::Mat& dst, const int position=200)
{
	cv::Mat src_copy = src.clone();
	cv::Mat dst_copy = cv::Mat(src.size(), src.type());

	// 创建指针指向src_copy的首地址
	uchar* srcData = src_copy.data;
	uchar* dstData = dst_copy.data;

	for (int j = 0; j < src_copy.cols; j++)
	{
		for (int i = 0; i < src_copy.rows; i++)
		{
			int gray = *(srcData + src_copy.step * i + src_copy.channels() * j + 0);

			if (j < position)
			{
				*(dstData + dst_copy.step * i + dst_copy.channels() * (src_copy.cols - position + j) + 0) = gray;
			}
			else
			{
				*(dstData + dst_copy.step * i + dst_copy.channels() * (j - position) + 0) = gray;
			}
		}
	}

	dst = dst_copy.clone();
	return 0;
}

#endif // !MYTRANSLATION_H
