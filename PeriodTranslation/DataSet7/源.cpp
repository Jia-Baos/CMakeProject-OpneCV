#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "Translation.h"
#include "Traverlsal.h"

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
	std::cout << "Version: " << CV_VERSION << std::endl;

	std::string template1_path = "D:/DataSet/dataset3/template/template1.jpg";
	std::string template2_path = "D:/DataSet/dataset3/template/template2.jpg";
	std::string src_path = "D:/DataSet/dataset7/data_croped/0000-001.jpg";

	std::string traversal_path = "D:/DataSet/dataset3/data_croped";
	std::string save_path = "D:/DataSet/dataset3/data_rgb/";

	cv::Mat src = cv::imread(src_path);
	cv::Mat template1 = cv::imread(template1_path);
	cv::Mat template2 = cv::imread(template2_path);

	// µ¥ÕÅÍ¼Ïñ²âÊÔ
	cv::Mat result2 = PeriodTranslation_Mode(src, template1, template2);
	cv::namedWindow("result2", cv::WINDOW_NORMAL);
	cv::imshow("result2", result2);

	//std::vector<std::string> fileNames;
	//getFileNames(traversal_path, fileNames);
	//for (const auto& img_path : fileNames)
	//{
	//	std::string img_name = img_path.substr(img_path.length() - 21, img_path.length());
	//	std::string abs_save_path = save_path + img_name;
	//	std::cout << img_path << std::endl;
	//	std::cout << abs_save_path << std::endl;

	//	//distance 1000
	//	cv::Mat dst = cv::imread(img_path);
	//	cv::Mat roi = dst(cv::Rect(0, 0, 3200, dst.rows));
	//	cv::Mat result = PeriodTranslation_Mode(roi, template1, template2);
	//	cv::imwrite(abs_save_path, result);
	//}

	cv::waitKey();
	return 0;
}