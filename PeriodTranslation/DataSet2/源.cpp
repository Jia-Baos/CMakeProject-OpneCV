#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Translation.h"
#include "Traverlsal.h"

int main(int argc, char* argv[])
{
	std::cout << "Version: " << CV_VERSION << std::endl;

	std::string traversal_path = "D:/DataSet/dataset3/data_croped";
	std::string save_path = "D:/DataSet/dataset3/data_translation/";
	std::string src_path = "D:/DataSet/dataset3/data_croped/0001-001.jpg";
	std::string template1_path = "D:/DataSet/dataset3/template/template1.jpg";
	std::string template2_path = "D:/DataSet/dataset3/template/template2.jpg";
	//std::string path = "D:\\Code-VS\\picture\\match-data\\graffiti.png";

	cv::Mat src = cv::imread(src_path);
	cv::Mat template1 = cv::imread(template1_path);
	cv::Mat template2 = cv::imread(template2_path);

	cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
	cv::cvtColor(template1, template1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(template2, template2, cv::COLOR_BGR2GRAY);

	//// ���� croped_distance
	//cv::Mat result = EdgeProcess(src);
	//cv::namedWindow("result", cv::WINDOW_NORMAL);
	//cv::imshow("result", result);

	// ����ͼ�����
	cv::Mat result = PeriodTranslation_Mode(src, template1, template2);
	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::imshow("result", result);

	//std::vector<std::string> fileNames;
	//getFileNames(traversal_path, fileNames);
	//for (const auto& img_path : fileNames)
	//{
	//	std::string img_name = img_path.substr(img_path.length() - 12, img_path.length());
	//	std::string abs_save_path = save_path + img_name;
	//	std::cout << img_path << std::endl;
	//	std::cout << abs_save_path << std::endl;

	//	//distance 400
	//	cv::Mat dst = cv::imread(img_path);
	//	cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
	//	cv::Mat result = PeriodTranslation_Mode(dst, template1, template2);
	//	cv::imwrite(abs_save_path, result);
	//}

	cv::waitKey();
	return 0;
}