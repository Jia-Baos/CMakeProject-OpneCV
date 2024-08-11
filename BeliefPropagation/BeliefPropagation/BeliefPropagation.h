#pragma once
#ifndef __BELIEFPROPAGATION_H__
#define __BELIEFPROPAGATION_H__

#include <iostream>
#include <opencv2/opencv.hpp>

#define L 16					//宏定义标记空间为16，即视差的最大取值范围
#define T 5						//宏定义迭代次数
#define C 25					//定义线性是函数的系数

struct Node_s {
	int x;						//表示行
	int y;						//表示列
	Node_s* left;				//左边节点
	Node_s* right;				//右边节点
	Node_s* up;					//上方节点
	Node_s* down;				//下方节点
	int messageUp[T][L];		//传给上方节点的消息向量
	int messageDown[T][L];		//传给下方节点的消息向量
	int messageLeft[T][L];		//传给左边节点的消息向量
	int messageRight[T][L];		//传给右边节点的消息向量
};


class BeliefPropagation
{
public:
	BeliefPropagation();
	virtual ~BeliefPropagation();

	// 构造函数，需要传入左视图和右视图
	BeliefPropagation(cv::Mat lImage, cv::Mat rImage);
	// 对节点初始化
	void initNode(Node_s* node, int i, int j);
	// 用链表表示图模型
	void generateList();
	// 用来测试链表的建立是否成功
	void printList();

	// 求解代价空间，x表示不同的视差，对于左视图(i,j)节点
	// 当x取不同视差，求解相似度(像素差值的绝对值)
	int Cost(int x, int li, int lj);

	// 辅助消息函数，计算能量的函数,从r节点传向e节点的信息
	void computeME(Node_s* r, Node_s* e, int t);
	// 计算消息矩阵，T为迭代次数
	void computeM();
	// 找到使得节点n置信度最小的视差x
	int  computeDisparityByBelief(Node_s* n);
	// 生成视差图
	void generateDisparity();

	// 释放存储空间
	void freeList();
	// BP算法的流程
	void doBp();
	cv::Mat getDisparity();
	
	// 视差图
	cv::Mat disparity;

private:
	// 表示构造的“图”的头结点
	// 以一张图左上方的像素为头结
	Node_s* head;
	cv::Mat leftImage;
	cv::Mat rightImage;
	cv::Mat leftGray;
	cv::Mat rightGray;
};

#endif // !__BELIEFPROPAGATION_H__
