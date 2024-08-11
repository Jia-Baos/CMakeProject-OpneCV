#include <iostream>
#include <string>
#include "Graph.h"


const int MAX_DISPARITY = 24;
const int ITERATION = 20;
const int LAMBDA = 10;
const int SMOOTHNESS_PARAM = 2;

double countTime() {
	return static_cast<double>(clock());
}

int main(int argc, char* argv[])
{
	const double beginTime = countTime();

	MarkovRandomField mrf;
	const std::string leftImgPath = "D:/Code-VS/picture/test-data/teddy/im2.png";
	const std::string rightImgPath = "D:/Code-VS/picture/test-data/teddy/im6.png";
	MarkovRandomFieldParam param;

	param.iteration = ITERATION;
	param.lambda = LAMBDA;
	param.maxDisparity = MAX_DISPARITY;
	param.smoothnessParam = SMOOTHNESS_PARAM;

	initializeMarkovRandomField(mrf, leftImgPath, rightImgPath, param);

	for (int i = 0; i < mrf.param.iteration; i++) {
		beliefPropagation(mrf, Left);
		beliefPropagation(mrf, Right);
		beliefPropagation(mrf, Up);
		beliefPropagation(mrf, Down);

		const energy_t energy = calculateMaxPosteriorProbability(mrf);

		std::cout << "Iteration: " << i << ";  Energy: " << energy << "." << std::endl;
	}

	cv::Mat output = cv::Mat::zeros(mrf.height, mrf.width, CV_8U);

	for (int i = mrf.param.maxDisparity; i < mrf.height - mrf.param.maxDisparity; i++) {
		for (int j = mrf.param.maxDisparity; j < mrf.width - mrf.param.maxDisparity; j++) {
			output.at<uchar>(i, j) = mrf.grid[i * mrf.width + j].bestAssignmentIndex * (256 / mrf.param.maxDisparity);
		}
	}

	const double endTime = countTime();

	std::cout << (endTime - beginTime) / CLOCKS_PER_SEC << std::endl;

	
	imshow("Output", output);
	cv::waitKey();
	return 0;
}