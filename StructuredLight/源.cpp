#include <iostream>
#include <opencv2/opencv.hpp>

#include "StructuredLight.h"

#ifdef _DEBUG
#pragma comment(lib, "D:/opencv_contrib/build/x64/vc17/lib/opencv_world460d.lib")
#else
#pragma comment(lib, "D:/opencv_contrib/build/x64/vc17/lib/opencv_world460.lib")
#endif // _DEBUG

int main(int argc, char* argv[])
{
	std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

	std::string img_light_path = "D:/Code-VS/Project-StructuredLight/David/1.bmp";
	std::string img_dark_path = "D:/Code-VS/Project-StructuredLight/David/2.bmp";
	std::string img1_gray_code_path = "D:/Code-VS/Project-StructuredLight/David/3.bmp";
	std::string img2_gray_code_path = "D:/Code-VS/Project-StructuredLight/David/4.bmp";
	std::string img3_gray_code_path = "D:/Code-VS/Project-StructuredLight/David/5.bmp";
	std::string img4_gray_code_path = "D:/Code-VS/Project-StructuredLight/David/6.bmp";
	std::string img5_gray_code_path = "D:/Code-VS/Project-StructuredLight/David/7.bmp";
	std::string img1_phsae_shift_path = "D:/Code-VS/Project-StructuredLight/David/8.bmp";
	std::string img2_phsae_shift_path = "D:/Code-VS/Project-StructuredLight/David/9.bmp";
	std::string img3_phsae_shift_path = "D:/Code-VS/Project-StructuredLight/David/10.bmp";
	std::string img4_phsae_shift_path = "D:/Code-VS/Project-StructuredLight/David/11.bmp";

	cv::Mat img_light = cv::imread(img_light_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img_dark = cv::imread(img_dark_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img1_gray_code = cv::imread(img1_gray_code_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img2_gray_code = cv::imread(img2_gray_code_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img3_gray_code = cv::imread(img3_gray_code_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img4_gray_code = cv::imread(img4_gray_code_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img5_gray_code = cv::imread(img5_gray_code_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img1_phase_shift = cv::imread(img1_phsae_shift_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img2_phase_shift = cv::imread(img2_phsae_shift_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img3_phase_shift = cv::imread(img3_phsae_shift_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img4_phase_shift = cv::imread(img4_phsae_shift_path, cv::IMREAD_GRAYSCALE);

	std::vector<cv::Mat> imgs_gray_code = { img1_gray_code, img2_gray_code,
		img3_gray_code, img4_gray_code, img5_gray_code };

	std::vector<cv::Mat> imgs_phase_shift = { img1_phase_shift, img2_phase_shift,
		img3_phase_shift, img4_phase_shift };

	GrayPhase* structured_light = new GrayPhase();
	structured_light->SetInput(img_light, img_dark,
		imgs_gray_code, imgs_phase_shift);
	structured_light->Compute();

	cv::namedWindow("unwrapped_phase", cv::WINDOW_NORMAL);
	cv::imshow("unwrapped_phase", structured_light->unwrapped_phase_dis_);
	cv::imwrite("unwrapped_phase.png", structured_light->unwrapped_phase_dis_);
	delete structured_light;

	cv::waitKey();
	return 0;
}
