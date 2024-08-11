#include "Translation.h"

void ComputeParameter(const cv::Mat& image,
	const cv::Mat& template1,
	const cv::Mat& template2,
	std::vector<double>& similarity,
	std::vector<cv::Point>& location)
{
	cv::Mat matTmp = image.clone();
	cv::Mat templateTmp1 = template1.clone();
	cv::Mat templateTmp2 = template2.clone();

	cv::cvtColor(matTmp, matTmp, cv::COLOR_BGR2GRAY);
	cv::cvtColor(templateTmp1, templateTmp1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(templateTmp2, templateTmp2, cv::COLOR_BGR2GRAY);

	cv::Mat matchResult1 = cv::Mat(matTmp.size(), CV_32FC1);
	cv::Mat matchResult2 = cv::Mat(matTmp.size(), CV_32FC1);

	cv::matchTemplate(matTmp, templateTmp1, matchResult1, cv::TM_CCOEFF_NORMED);
	double minVal1, maxVal1;
	cv::Point minLoc1, maxLoc1;
	cv::minMaxLoc(matchResult1, &minVal1, &maxVal1, &minLoc1, &maxLoc1, cv::Mat());

	cv::matchTemplate(matTmp, templateTmp2, matchResult2, cv::TM_CCOEFF_NORMED);
	double minVal2, maxVal2;
	cv::Point minLoc2, maxLoc2;
	cv::minMaxLoc(matchResult2, &minVal2, &maxVal2, &minLoc2, &maxLoc2, cv::Mat());

	similarity.push_back(maxVal1);
	similarity.push_back(maxVal2);
	location.push_back(maxLoc1);
	location.push_back(maxLoc2);

	std::cout << "Template1: " << maxVal1 << ", " << maxLoc1 << std::endl;
	std::cout << "Template2: " << maxVal2 << ", " << maxLoc2 << std::endl;

	//// 实际匹配结果展示
	//cv::rectangle(matTmp, cv::Point(maxLoc1.x, maxLoc1.y),
	//	cv::Point(maxLoc1.x + templateTmp1.cols, maxLoc1.y + templateTmp1.rows),
	//	cv::Scalar(0, 0, 0), 4, 1, 0);
	//cv::rectangle(matTmp, cv::Point(maxLoc2.x, maxLoc2.y),
	//	cv::Point(maxLoc2.x + templateTmp2.cols, maxLoc2.y + templateTmp2.rows),
	//	cv::Scalar(0, 0, 0), 4, 1, 0);
	//cv::namedWindow("matTmp", cv::WINDOW_NORMAL);
	//cv::imshow("matTmp", matTmp);
}

void EdgeProcess(const cv::Mat& src, cv::Mat& dst)
{
	cv::Mat rect1 = src(cv::Rect(src.cols - 100, 0, 100, src.rows));
	cv::Mat rect2 = src(cv::Rect(0, 0, src.cols / 2, src.rows));

	cv::cvtColor(rect1, rect1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(rect2, rect2, cv::COLOR_BGR2GRAY);

	cv::Mat matchResult = cv::Mat(rect2.size(), CV_32FC1);
	cv::matchTemplate(rect2, rect1, matchResult, cv::TM_CCOEFF_NORMED);
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	/*cv::rectangle(rect2, cv::Point(maxLoc.x, maxLoc.y),
		cv::Point(maxLoc.x + rect1.cols, maxLoc.y + rect1.rows),
		cv::Scalar(0, 0, 0), 4, 1, 0);
	cv::namedWindow("rect1", cv::WINDOW_NORMAL);
	cv::imshow("rect1", rect1);
	cv::namedWindow("rect2", cv::WINDOW_NORMAL);
	cv::imshow("rect2", rect2);*/

	std::cout << "The distance should be croped: " << maxLoc.x + rect1.cols << std::endl;
	dst = src(cv::Rect(maxLoc.x, 0, src.cols - maxLoc.x - rect1.cols, src.rows));
}

void PeriodTranslation_BGR(const cv::Mat& src, cv::Mat& dst, const int distance)
{
	cv::Mat srctmp = src.clone();
	dst = cv::Mat(srctmp.size(), srctmp.type());

	std::cout << srctmp.size() << "; " << srctmp.channels() << std::endl;
	std::cout << dst.size() << "; " << dst.channels() << std::endl;

	// 创建指针指向src的首地址
	uchar* srcData = srctmp.data;
	uchar* dstData = dst.data;
	const int step = srctmp.step[0];

	for (int j = 0; j < srctmp.cols; j++)
	{
		for (int i = 0; i < srctmp.rows; i++)
		{
			int b = *(srcData + step * i + srctmp.channels() * j + 0);
			int g = *(srcData + step * i + srctmp.channels() * j + 1);
			int r = *(srcData + step * i + srctmp.channels() * j + 2);

			if (j < distance)
			{
				*(dstData + step * i + dst.channels() * (dst.cols - distance + j) + 0) = b;
				*(dstData + step * i + dst.channels() * (dst.cols - distance + j) + 1) = g;
				*(dstData + step * i + dst.channels() * (dst.cols - distance + j) + 2) = r;
			}
			else
			{
				*(dstData + step * i + dst.channels() * (j - distance) + 0) = b;
				*(dstData + step * i + dst.channels() * (j - distance) + 1) = g;
				*(dstData + step * i + dst.channels() * (j - distance) + 2) = r;
			}
		}
	}
}

cv::Mat PeriodTranslation_Mode(const cv::Mat& image, const cv::Mat& template1, const cv::Mat& template2)
{
	int croped_distance = 200;
	cv::Mat image_croped, image_croped_copy;
	std::vector<double> similarityTmp;
	std::vector<cv::Point> locationTmp;

	// 首次计算相似度，判断如何进行裁剪
	ComputeParameter(image, template1, template2, similarityTmp, locationTmp);

	if (similarityTmp[0] >= 0.6 && similarityTmp[1] >= 0.3)
	{
		std::cout << "The target of template1 and template2 are comlete..." << std::endl;
		image_croped = image(cv::Rect(0, 0, image.cols - croped_distance, image.rows));
	}
	else
	{
		std::cout << "The target of template1 and template2 were divided into two parts..." << std::endl;
		EdgeProcess(image, image_croped);
	}

	// 对裁剪后的图像循环进行周期平移，当置信度达到给定值时停止
	while (similarityTmp[0] < 0.5 || similarityTmp[1] < 0.3)
	{
		std::cout << "iterating..." << std::endl;
		PeriodTranslation_BGR(image_croped, image_croped_copy, 100);
		similarityTmp.clear();
		locationTmp.clear();
		ComputeParameter(image_croped_copy, template1, template2, similarityTmp, locationTmp);
		image_croped = image_croped_copy.clone();
	}

	// 此时 template1 和 template2 的置信度都已满足要求，则进行最后一次平移以对齐区域
	PeriodTranslation_BGR(image_croped, image_croped_copy, locationTmp[0].x - 100);
	return image_croped_copy;
}
