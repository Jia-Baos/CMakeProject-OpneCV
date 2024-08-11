#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/core/utils/logger.hpp>

#ifdef _DEBUG
#pragma comment(lib, "D:/opencv_contrib/opencv_contrib/x64/vc17/lib/opencv_world460d.lib")
#else
#pragma comment(lib, "D:/opencv_contrib/opencv_contrib/x64/vc17/lib/opencv_world460.lib")
#endif // _DEBUG

int main(int argc, char* argv[])
{
	std::cout << "Version: " << CV_VERSION << std::endl;
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	std::string path = "D:/Code-VS/picture/OpticalFlow/frame_0017.png";
	//std::string path = "D:/Code-VS/picture/figs/test4/dataset8-3.png";

	cv::Mat src = cv::imread(path);
	cv::Mat src_cpy = src.clone();
	cv::GaussianBlur(src, src, cv::Size(3, 3), 0.6);
	cv::cvtColor(src, src, cv::COLOR_BGR2Lab);

	cv::Mat mask, label;
	cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(src, cv::ximgproc::SLIC);
	// cv::Ptr<cv::ximgproc::SuperpixelLSC> slic = cv::ximgproc::createSuperpixelLSC(src, cv::ximgproc::SLIC);
	slic->iterate(10);							// 迭代次数
	slic->enforceLabelConnectivity();			// superpixel的下限
	slic->getLabelContourMask(mask);			// 超像素分割的mask
	slic->getLabels(label);						// 超像素分割的label
	int num = slic->getNumberOfSuperpixels();	// 超像素的数目

	std::cout << "num: " << num << std::endl;

	cv::Mat mask_inv, dst;
	cv::bitwise_not(mask, mask_inv);
	cv::bitwise_and(src_cpy, src_cpy, dst, mask_inv);

	cv::namedWindow("dst", cv::WINDOW_NORMAL);
	cv::imshow("dst", dst);

	cv::waitKey();
	return 0;
}
