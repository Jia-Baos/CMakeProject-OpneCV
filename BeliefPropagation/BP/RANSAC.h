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
	int max_iter_;							// ����������
	float threshold_;						// �ڵ���ֵ
	std::vector<cv::Point3f> points_3d_;	// ����ĵ㼯					
	Plane plane_best_;						// ��ϵ�ƽ�����
public:
	RANSAC();
	virtual ~RANSAC();
	virtual void SetInput(const std::vector<cv::Point3f>& points_3d);
	virtual void Compute(const std::vector<cv::Point3f>& points_3d);
};
#endif // !__RANSAC_H__
