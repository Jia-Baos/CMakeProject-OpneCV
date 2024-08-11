#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "./GeodesicDistance.hpp"
#include "./Saliency.hpp"


int main(int argc, char* argv[])
{
	std::cout << "Version: " << CV_VERSION << std::endl;
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	std::string image_path = "D:/Code-VS/CMakeProject-GeodesicDistance/33.jpg";
	std::string match_path = "D:/Code-VS/CMakeProject-GeodesicDistance/frame_0029.txt";
	std::string model_path = "D:/Code-VS/CMakeProject-GeodesicDistance/model.yml";

	cv::Mat image = cv::imread(image_path);

	// Read matches
	std::cout << "Read seeds..." << std::endl;
	std::vector<cv::Point2f> seeds;

	/*FILE* file;
	float x1 = 0.0;
	float y1 = 0.0;
	float x2 = 0.0;
	float y2 = 0.0;
	fopen_s(&file, match_path.c_str(), "r");
	while (!feof(file) &&
		fscanf_s(file, "%f %f %f %f%*[^\n]", &x1, &y1, &x2, &y2) == 4)
	{
		seeds.emplace_back(cv::Point2f(x1, y1));
	}*/

	seeds.emplace_back(cv::Point2f(148, 317));
	seeds.emplace_back(cv::Point2f(257, 314));
	seeds.emplace_back(cv::Point2f(389, 309));
	seeds.emplace_back(cv::Point2f(541, 304));
	seeds.emplace_back(cv::Point2f(688, 305));

	//// Remove matches coming from a pixel with a low saliency
	//cv::Mat image_saliency = image.clone();
	//cv::cvtColor(image_saliency, image_saliency, cv::COLOR_BGR2Lab);
	//image_saliency.convertTo(image_saliency, CV_32FC3);

	//cv::Mat saliency_map = cv::Mat::zeros(image_saliency.size(), CV_32FC1);
	//Saliency(image_saliency, saliency_map, 0.8, 1.0);
	//std::vector<cv::Point2f> seeds_reserved;
	//for (auto& iter : seeds)
	//{
	//	const int x = iter.y;
	//	const int y = iter.x;
	//	if (saliency_map.ptr<float>(x)[y] > 0.45f)
	//	{
	//		seeds_reserved.emplace_back(iter);
	//	}
	//}

	// Compute cost map by SED
	std::cout << "Read edges..." << std::endl;
	cv::Mat image_sed = image.clone();
	image_sed.convertTo(image_sed, CV_32FC3, 1 / 255.0);
	cv::Mat edges_map = cv::Mat::zeros(image_sed.size(), CV_32FC1);

	cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar =
		cv::ximgproc::createStructuredEdgeDetection(model_path);
	pDollar->detectEdges(image_sed, edges_map);

	cv::normalize(edges_map, edges_map, 0, 1,
		cv::NORM_MINMAX, -1, cv::Mat());
	edges_map.convertTo(edges_map, CV_32FC1, 255.0);

	cv::Mat dist_map;
	std::chrono::steady_clock::time_point code_in = std::chrono::steady_clock::now();
	GetDMRasterscan(edges_map, dist_map, seeds, 8, 1.0, geodesic);
	std::chrono::steady_clock::time_point code_out = std::chrono::steady_clock::now();

	double spend_time = std::chrono::duration<double>(code_out - code_in).count();
	std::cout << "spend_time: " << spend_time << " s" << std::endl;

	cv::normalize(dist_map, dist_map, 0, 255,
		cv::NORM_MINMAX, -1, cv::Mat());
	dist_map.convertTo(dist_map, CV_8UC1);
	cv::applyColorMap(dist_map, dist_map, cv::COLORMAP_JET);

	// 绘制种子点
	for (auto& iter : seeds)
	{
		cv::circle(dist_map, cv::Point2f(iter.x, iter.y), 3, cv::Scalar(0, 0, 255), -1);
	}

	cv::namedWindow("dist_map", cv::WINDOW_NORMAL);
	cv::imshow("dist_map", dist_map);
	cv::waitKey();

	return 0;
}
