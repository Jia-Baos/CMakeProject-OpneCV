#pragma once
#ifndef __BELIEFPROPAGATION_H__
#define __BELIEFPROPAGATION_H__

#include <iostream>
#include <opencv2/opencv.hpp>

const int LABELS = 50;				// THe number of offset
const int Iteration = 20;
const int LAMBDA = 10;
const int SMOOTHNESS_TRUC = 2;
const int OFFSET_MIN = -50;		
const int STEP = 2;

enum ATTRIBUTE { LEFT, RIGHT, UP, DOWN, DATA };
enum DIRECTION {HORINZONTAL, VERTICAL};

struct MarkovRandomFieldNode {
	unsigned int msg[5][LABELS];
	int best_assignment;
};


struct MarkovRandomField {
	std::vector<MarkovRandomFieldNode> grid;
	int width, height;
};

class BeliefPropagation
{
private:
	int width_blocks_;							// block的宽度
	int width_search_;							// 搜索区域的宽度，block区域向外填充的宽度
	cv::Mat fixed_image_, moved_image_;			// 拷贝原始图像
	int col_nums_;
	int row_nums_;

public:
	BeliefPropagation();

	virtual ~BeliefPropagation();
	
	virtual bool ImagePadded(const cv::Mat& input1,
		const cv::Mat& input2,
		cv::Mat& output1,
		cv::Mat& output2);

	virtual bool SetInput(const cv::Mat& fixed_image,
		const cv::Mat& moved_image);

	virtual bool Compute(const cv::Mat& fixed_image,
		const cv::Mat& moved_image);

	virtual void InitialMarkovRandomField(const cv::Mat& fixed_image_current,
		const cv::Mat& moved_image_current,
		const cv::Mat& blocks_offset,
		MarkovRandomField& mrf_x,
		MarkovRandomField& mrf_y);

	virtual unsigned int DataCost(const cv::Mat& fixed_image_current,
		const cv::Mat& moved_image_current,
		int i, int j, int offset,
		DIRECTION direction);

	virtual unsigned int SmoothCost(int i, int j);

	virtual void BP(MarkovRandomField& mrf,
		ATTRIBUTE attribute);

	virtual void SendMsg(MarkovRandomField& mrf,
		int j, int i,
		ATTRIBUTE attribute);

	virtual unsigned int MAP(MarkovRandomField& mrf);

};


#endif // !__BELIEFPROPAGATION_H__
