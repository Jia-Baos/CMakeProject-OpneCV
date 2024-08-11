#pragma once
#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

enum Direction { Left, Right, Up, Down, Data };

typedef unsigned int msg_t;
typedef unsigned int energy_t;
typedef unsigned int smoothness_cost_t;
typedef unsigned int data_cost_t;

struct MarkovRandomFieldNode {
	msg_t* leftMessage;
	msg_t* rightMessage;
	msg_t* upMessage;
	msg_t* downMessage;
	msg_t* dataMessage;
	int bestAssignmentIndex;
};

struct MarkovRandomFieldParam {
	int maxDisparity;
	int lambda;
	int iteration;
	int smoothnessParam;
};

struct MarkovRandomField {
	std::vector<MarkovRandomFieldNode> grid;
	MarkovRandomFieldParam param;
	int height, width;
};

void initializeMarkovRandomField(MarkovRandomField& mrf, std::string leftImgPath, std::string rightImgPath,
	MarkovRandomFieldParam param);
void sendMsg(MarkovRandomField& mrf, int x, int y, Direction dir);
void beliefPropagation(MarkovRandomField& mrf, Direction dir);

data_cost_t calculateDataCost(cv::Mat& leftImg, cv::Mat& rightImg, int x, int y, int disparity);
smoothness_cost_t calculateSmoothnessCost(int i, int j, int lambda, int smoothnessParam);
energy_t calculateMaxPosteriorProbability(MarkovRandomField& mrf);

#endif // !GRAPH_H
