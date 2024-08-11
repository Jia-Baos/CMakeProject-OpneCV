#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "BeliefPropagation.h"
#include "RANSAC.h"

int main(int argc, char* argv[])
{
	std::cout << "Version: " << CV_VERSION << std::endl;
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	std::string fixed_image_path = "D:\\Code-VS\\picture\\data\\fixed_image1.jpg";
	std::string moved_image_path = "D:\\Code-VS\\picture\\data\\moved_image1.jpg";
	//std::string fixed_image_path = "D:\\Code-VS\\picture\\data\\dataset7-test1.jpg";
	//std::string moved_image_path = "D:\\Code-VS\\picture\\data\\dataset7-test2.jpg";
	//std::string fixed_image_path = "D:\\DataSet\\dataset3\\template\\template1.bmp";
	//std::string moved_image_path = "D:\\DataSet\\dataset3\\template\\template5.bmp";

	cv::Mat fixed_image = cv::imread(fixed_image_path);
	cv::Mat moved_image = cv::imread(moved_image_path);

	BeliefPropagation* bp = new BeliefPropagation();
	bp->Compute(fixed_image, moved_image);
	// visit member variables of class through friend function
	delete bp;

	/*cv::Point3f point_3d_1 = cv::Point3f(0, 0, 1);
	cv::Point3f point_3d_2 = cv::Point3f(0, 1, 0);
	cv::Point3f point_3d_3 = cv::Point3f(1, 0, 0);
	cv::Point3f point_3d_4 = cv::Point3i(1, 0, 0.1);
	cv::Point3f point_3d_5 = cv::Point3i(0, 1, 0.1);
	std::vector<cv::Point3f> point_3d;
	point_3d.push_back(point_3d_1);
	point_3d.push_back(point_3d_2);
	point_3d.push_back(point_3d_3);
	point_3d.push_back(point_3d_4);
	point_3d.push_back(point_3d_5);
	RANSAC* ransac = new RANSAC();
	ransac->Compute(point_3d);
	delete ransac;*/

	cv::waitKey();
	return 0;
}