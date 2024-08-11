#pragma once
#ifndef __BELIEFPROPAGATION_H__
#define __BELIEFPROPAGATION_H__

#include <iostream>
#include <opencv2/opencv.hpp>

#define L 16					//�궨���ǿռ�Ϊ16�����Ӳ�����ȡֵ��Χ
#define T 5						//�궨���������
#define C 25					//���������Ǻ�����ϵ��

struct Node_s {
	int x;						//��ʾ��
	int y;						//��ʾ��
	Node_s* left;				//��߽ڵ�
	Node_s* right;				//�ұ߽ڵ�
	Node_s* up;					//�Ϸ��ڵ�
	Node_s* down;				//�·��ڵ�
	int messageUp[T][L];		//�����Ϸ��ڵ����Ϣ����
	int messageDown[T][L];		//�����·��ڵ����Ϣ����
	int messageLeft[T][L];		//������߽ڵ����Ϣ����
	int messageRight[T][L];		//�����ұ߽ڵ����Ϣ����
};


class BeliefPropagation
{
public:
	BeliefPropagation();
	virtual ~BeliefPropagation();

	// ���캯������Ҫ��������ͼ������ͼ
	BeliefPropagation(cv::Mat lImage, cv::Mat rImage);
	// �Խڵ��ʼ��
	void initNode(Node_s* node, int i, int j);
	// �������ʾͼģ��
	void generateList();
	// ������������Ľ����Ƿ�ɹ�
	void printList();

	// �����ۿռ䣬x��ʾ��ͬ���Ӳ��������ͼ(i,j)�ڵ�
	// ��xȡ��ͬ�Ӳ������ƶ�(���ز�ֵ�ľ���ֵ)
	int Cost(int x, int li, int lj);

	// ������Ϣ���������������ĺ���,��r�ڵ㴫��e�ڵ����Ϣ
	void computeME(Node_s* r, Node_s* e, int t);
	// ������Ϣ����TΪ��������
	void computeM();
	// �ҵ�ʹ�ýڵ�n���Ŷ���С���Ӳ�x
	int  computeDisparityByBelief(Node_s* n);
	// �����Ӳ�ͼ
	void generateDisparity();

	// �ͷŴ洢�ռ�
	void freeList();
	// BP�㷨������
	void doBp();
	cv::Mat getDisparity();
	
	// �Ӳ�ͼ
	cv::Mat disparity;

private:
	// ��ʾ����ġ�ͼ����ͷ���
	// ��һ��ͼ���Ϸ�������Ϊͷ��
	Node_s* head;
	cv::Mat leftImage;
	cv::Mat rightImage;
	cv::Mat leftGray;
	cv::Mat rightGray;
};

#endif // !__BELIEFPROPAGATION_H__
