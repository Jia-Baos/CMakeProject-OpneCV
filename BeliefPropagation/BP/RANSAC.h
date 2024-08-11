#pragma once
#ifndef __RANSAC_H__
#define __RANSAC_H__

#include <iostream>
#include <vector>
#include <ctime>
#include <opencv2/opencv.hpp>

struct Plane
{
	float a;
	float b;
	float c;
	float d;
};

class RANSAC
{
private:
	int max_iter_;							// 最大迭代次数
	float threshold_;						// 内点阈值
	std::vector<cv::Point3f> points_3d_;	// 输入的点集					
	Plane plane_best_;						// 拟合的平面参数
public:
	RANSAC();
	virtual ~RANSAC();
	virtual void SetInput(const std::vector<cv::Point3f>& points_3d);
	virtual void Compute(const std::vector<cv::Point3f>& points_3d);
};
#endif // !__RANSAC_H__
