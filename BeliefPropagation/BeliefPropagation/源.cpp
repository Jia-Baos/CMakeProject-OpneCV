#include <iostream>
#include <string>
#include "BeliefPropagation.h"

int main(int argc, char* argv[])
{
	const std::string left_image_path = "D:/Code-VS/picture/test-data/cones/im0.png";
	const std::string right_image_path = "D:/Code-VS/picture/test-data/cones/im1.png";

	cv::Mat left_image = cv::imread(left_image_path);
	cv::Mat right_image = cv::imread(right_image_path);
	cv::cvtColor(left_image, left_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(right_image, right_image, cv::COLOR_BGR2GRAY);

	BeliefPropagation* bp = new BeliefPropagation(left_image, right_image);
	bp->generateList();
	bp->computeM();
	bp->generateDisparity();
	delete bp;

	cv::namedWindow("disparity", cv::WINDOW_NORMAL);
	cv::imshow("disparity", bp->disparity);
	
	cv::waitKey();
	return 0;
}