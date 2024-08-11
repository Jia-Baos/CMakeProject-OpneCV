#include "Translation.h"

bool ComputeParameter(const cv::Mat& image, const cv::Mat& template1, const cv::Mat& template2,
	std::vector<double>& similarity, std::vector<cv::Point>& location)
{
	cv::Mat matTmp = image.clone();
	cv::Mat templateTmp1 = template1.clone();
	cv::Mat templateTmp2 = template2.clone();

	cv::Mat matchResult1 = cv::Mat(matTmp.size(), matTmp.type());
	cv::Mat matchResult2 = cv::Mat(matTmp.size(), matTmp.type());

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

	return 0;
}

cv::Mat PeriodTranslation_Mode(const cv::Mat& image, const cv::Mat& template1, const cv::Mat& template2)
{
	cv::Mat image_tmp, template1_tmp, template2_tmp;
	cv::cvtColor(image, image_tmp, cv::COLOR_BGR2GRAY);
	cv::cvtColor(template1, template1_tmp, cv::COLOR_BGR2GRAY);
	cv::cvtColor(template2, template2_tmp, cv::COLOR_BGR2GRAY);

	int croped_distance = 400;
	cv::Mat matTmp_croped;
	cv::Mat matTmp = image.clone();
	std::vector<double> similarityTmp;
	std::vector<cv::Point> locationTmp;

	// 首次计算相似度，判断如何进行裁剪
	ComputeParameter(matTmp, template1, template2, similarityTmp, locationTmp);

	if (similarityTmp[0] >= 0.6 && similarityTmp[1] >= 0.6)
	{
		std::cout << "The target of template1 and template2 are comlete..." << std::endl;
		matTmp_croped = matTmp(cv::Rect(0, 0, matTmp.cols - croped_distance, matTmp.rows));
	}
	else
	{
		std::cout << "The target of template1 and template2 were divided into two parts..." << std::endl;
		matTmp_croped = EdgeProcess(matTmp);
	}

	// 对裁剪后的图像循环进行周期平移，当置信度达到给定值时停止
	while (similarityTmp[0] < 0.6 || similarityTmp[1] < 0.6)
	{
		std::cout << "iterating..." << std::endl;
		matTmp_croped = PeriodTranslation_GRAY(matTmp_croped, 100);
		similarityTmp.clear();
		locationTmp.clear();
		ComputeParameter(matTmp_croped, template1, template2, similarityTmp, locationTmp);
	}

	// 此时 template1 和 template2 的置信度都已满足要求，则进行最后一次平移以对齐区域
	matTmp_croped = PeriodTranslation_BGR(matTmp_croped, locationTmp[0].x - 100);
	return matTmp_croped;
}

cv::Mat EdgeProcess(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	cv::Mat rect1 = matTmp(cv::Rect(matTmp.cols - 100, 0, 100, matTmp.rows));
	cv::Mat rect2 = matTmp(cv::Rect(0, 0, matTmp.cols / 2, matTmp.rows));

	cv::Mat matchResult = cv::Mat(rect2.size(), rect2.type());
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
	cv::Mat result = matTmp(cv::Rect(maxLoc.x, 0, matTmp.cols - maxLoc.x - rect1.cols, matTmp.rows));

	return result;
}

cv::Mat PeriodTranslation_BGR(const cv::Mat& src, const int distance)
{
	cv::Mat matTmp = src.clone();
	cv::Mat dst = cv::Mat(matTmp.size(), matTmp.type());

	std::cout << matTmp.size() << "; " << matTmp.channels() << std::endl;
	std::cout << dst.size() << "; " << dst.channels() << std::endl;

	// 创建指针指向matTmp的首地址
	uchar* srcData = matTmp.data;
	uchar* dstData = dst.data;
	const int step = matTmp.step[0] / sizeof(srcData[0]);

	for (int j = 0; j < matTmp.cols; j++)
	{
		for (int i = 0; i < matTmp.rows; i++)
		{
			int b = *(srcData + step * i + matTmp.channels() * j + 0);
			int g = *(srcData + step * i + matTmp.channels() * j + 1);
			int r = *(srcData + step * i + matTmp.channels() * j + 2);

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
	return dst;
}

cv::Mat PeriodTranslation_GRAY(const cv::Mat& src, const int distance)
{
	cv::Mat matTmp = src.clone();
	cv::Mat dst = cv::Mat::zeros(matTmp.size(), matTmp.type());

	// 创建指针指向matTmp的首地址
	uchar* srcData = matTmp.data;
	uchar* dstData = dst.data;
	const int step = matTmp.step[0] / sizeof(srcData[0]);

	for (int j = 0; j < matTmp.cols; j++)
	{
		for (int i = 0; i < matTmp.rows; i++)
		{
			int gray = *(srcData + step * i + matTmp.channels() * j + 0);

			if (j < distance)
			{
				*(dstData + step * i + dst.channels() * (matTmp.cols - distance + j) + 0) = gray;
			}
			else
			{
				*(dstData + step * i + dst.channels() * (j - distance) + 0) = gray;
			}
		}
	}
	return dst;
}