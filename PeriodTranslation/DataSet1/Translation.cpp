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
	//	cv::Scalar(255, 255, 255), 4, 1, 0);
	//cv::rectangle(matTmp, cv::Point(maxLoc2.x, maxLoc2.y),
	//	cv::Point(maxLoc2.x + templateTmp2.cols, maxLoc2.y + templateTmp2.rows),
	//	cv::Scalar(255, 255, 255), 4, 1, 0);
	//cv::namedWindow("matTmp", cv::WINDOW_NORMAL);
	//cv::imshow("matTmp", matTmp);

	return 0;
}

cv::Mat PeriodTranslation_Mode(const cv::Mat& image, const cv::Mat& template1, const cv::Mat& template2)
{
	cv::Mat matTmp = cv::Mat::zeros(image.size(), CV_8UC1);
	cv::Mat template1Tmp = cv::Mat::zeros(template1.size(), CV_8UC1);
	cv::Mat template2Tmp = cv::Mat::zeros(template2.size(), CV_8UC1);
	if (image.channels() == 3 || template1.channels() == 3 || template2.channels() == 3)
	{
		cv::cvtColor(image, matTmp, cv::COLOR_BGR2GRAY);
		cv::cvtColor(template1, template1Tmp, cv::COLOR_BGR2GRAY);
		cv::cvtColor(template2, template2Tmp, cv::COLOR_BGR2GRAY);
	}
	else
	{
		matTmp = image.clone();
		template1Tmp = template1Tmp.clone();
		template2Tmp = template2Tmp.clone();
	}

	// 首次计算相似度，判断如何进行裁剪
	cv::Mat matTmp_croped;
	int croped_distance = 150;
	std::vector<double> similarityTmp;
	std::vector<cv::Point> locationTmp;
	ComputeParameter(matTmp, template1Tmp, template2Tmp, similarityTmp, locationTmp);

	if (similarityTmp[0] >= 0.5 && similarityTmp[1] >= 0.5)
	{
		// 此时直接裁剪一部分进行平移
		std::cout << "The target of template1 and template2 are comlete..." << std::endl;
		matTmp_croped = matTmp(cv::Rect(0, 0, matTmp.cols - croped_distance, matTmp.rows));
	}
	else
	{
		if (similarityTmp[0] < 0.4)
		{
			std::cout << "The target of template1 weas divided into two parts..." << std::endl;
			matTmp_croped = EdgeProcess(matTmp);
		}

		if (similarityTmp[1] < 0.4)
		{
			std::cout << "The target of template2 weas divided into two parts..." << std::endl;
			matTmp_croped = EdgeProcessEnhancement(matTmp);
		}
	}

	/*std::cout << matTmp_croped.size() << std::endl;
	std::cout << matTmp_croped.channels() << std::endl;
	std::cout << matTmp_croped.type() << std::endl;
	cv::namedWindow("matTmp1", cv::WINDOW_NORMAL);
	cv::imshow("matTmp1", matTmp_croped);*/

	// 对裁剪后的图像循环进行周期平移，当置信度达到给定值时停止
	while (similarityTmp[0] < 0.45 || similarityTmp[1] < 0.45)
	{
		std::cout << "iterating..." << std::endl;
		matTmp_croped = PeriodTranslation_GRAY(matTmp_croped, 100);
		similarityTmp.clear();
		locationTmp.clear();
		ComputeParameter(matTmp_croped, template1Tmp, template2Tmp, similarityTmp, locationTmp);
	}

	// 此时 template1 和 template2 的置信度都已满足要求，则进行最后一次平移以对齐区域
	matTmp_croped = PeriodTranslation_GRAY(matTmp_croped, locationTmp[0].x - 100);

	return matTmp_croped;
}

cv::Mat EdgeProcess(const cv::Mat& image)
{
	cv::Mat matTmp;
	cv::Mat template1Tmp, template2Tmp;
	if (image.channels() == 3)
	{
		cv::cvtColor(image, matTmp, cv::COLOR_BGR2GRAY);
	}
	else
	{
		matTmp = image.clone();
	}

	cv::Mat rect1 = matTmp(cv::Rect(0, 0, 100, matTmp.rows));
	cv::Mat rect2 = matTmp(cv::Rect(matTmp.cols - 200, 0, 200, matTmp.rows));

	cv::Mat matchResult = cv::Mat(rect2.size(), rect2.type());
	cv::matchTemplate(rect2, rect1, matchResult, cv::TM_CCOEFF_NORMED);
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	/*cv::rectangle(rect2, cv::Point(maxLoc.x, maxLoc.y),
		cv::Point(maxLoc.x + rect1.cols, maxLoc.y + rect1.rows),
		cv::Scalar(255, 255, 255), 4, 1, 0);
	cv::namedWindow("rect1", cv::WINDOW_NORMAL);
	cv::imshow("rect1", rect1);
	cv::namedWindow("rect2", cv::WINDOW_NORMAL);
	cv::imshow("rect2", rect2);*/

	std::cout << "The distance should be croped: " << maxLoc.x + rect1.cols << std::endl;
	cv::Mat result = matTmp(cv::Rect(0, 0, matTmp.cols - (rect2.cols - maxLoc.x), matTmp.rows));

	return result;
}

bool OffsetRevise(const std::vector<int>& src, std::vector<int>& dst)
{
	// 离群值的表现形式为偏离正常值过多，所以直接sort原vector，则离群值会分布在正常值的左右两侧
	// 再比较相邻元素的阈值即可筛选出正常值
	std::vector<int> src_sorted;
	src_sorted.assign(src.begin(), src.end());
	dst.assign(src.begin(), src.end());
	std::sort(src_sorted.begin(), src_sorted.end());

	int iter1, iter2;
	if (src_sorted.size() % 2 == 0)
	{
		iter1 = (src_sorted.size() / 2) - 1;
		iter2 = src_sorted.size() / 2;
	}
	else
	{
		iter1 = std::floor(src_sorted.size() / 2);
		iter2 = std::floor(src_sorted.size() / 2);
	}

	int left_bad_value = 0, right_bad_value = INT_MAX;
	bool left_bad_flag = false, right_bad_flag = false;
	while (iter1 != 0 || iter2 != src_sorted.size() - 1)
	{
		int temp1 = src_sorted[iter1];
		int temp2 = src_sorted[iter2];

		if (abs(src_sorted[--iter1] - temp1) < 5)
		{
			temp1 = src_sorted[iter1];
		}
		else
		{
			left_bad_value = src_sorted[iter1];
			left_bad_flag = true;
		}
		if (abs(src_sorted[++iter2] - temp2) < 5)
		{
			temp2 = src_sorted[iter2];
		}
		else
		{
			right_bad_value = src_sorted[iter2];
			right_bad_flag = true;
		}
	}

	// 对异常值进行插值，这里假设异常值不会出现在首尾
	for (int i = 1; i < dst.size() - 1; i++)
	{
		if (left_bad_flag && dst[i] <= left_bad_value)
		{
			dst[i] = std::floor((dst[i - 1] + dst[i + 1]) / 2);
		}
		if (right_bad_flag && dst[i] >= right_bad_value)
		{
			dst[i] = std::floor((dst[i - 1] + dst[i + 1]) / 2);
		}
	}

	return 0;
}

cv::Mat EdgeProcessEnhancement(const cv::Mat& image)
{
	// 所谓的增强就是进行逐行的精确匹配，之后再进行裁剪
	cv::Mat matTmp;
	cv::Mat template1Tmp, template2Tmp;
	if (image.channels() == 3)
	{
		cv::cvtColor(image, matTmp, cv::COLOR_BGR2GRAY);
	}
	else
	{
		matTmp = image.clone();
	}

	// 利用膨胀操作消除字符之间的间隙
	// 获取两个被切开区域的起始坐标和终止坐标
	// 此处代码作用有待商榷，后面已经找到匹配的区域，可以直接利用临近两个偏移量插值出来
	// 还是有必要，别切除的子区域大小可能不一致，此类情况较少出现，可忽略 

	// 从左右边缘根据水平投影结果切割子区域
	std::vector<int> rect1_regionIndex_temp, rect2_regionIndex_temp;
	cv::Mat rect1 = matTmp(cv::Rect(0, 0, 100, matTmp.rows));
	cv::Mat rect2 = matTmp(cv::Rect(matTmp.cols - 2 * 100, 0, 200, matTmp.rows));
	GetHorProjRegions(rect1, rect1_regionIndex_temp);
	GetHorProjRegions(rect2, rect2_regionIndex_temp);

	std::cout << "rect1_regionIndex_temp.size: " << rect1_regionIndex_temp.size() << std::endl;
	std::cout << "rect2_regionIndex_temp.size: " << rect2_regionIndex_temp.size() << std::endl;

	// 受到缺陷的影响，子区域数量会产生波动，可以采用元素配对的方法从中筛选出相匹配的子区域
	std::vector<int>::iterator iter1, iter2;
	iter1 = rect1_regionIndex_temp.begin();
	iter2 = rect2_regionIndex_temp.begin();
	std::vector<int> rect1_regionIndex, rect2_regionIndex;
	while (iter1 != rect1_regionIndex_temp.end() || iter2 != rect2_regionIndex_temp.end())
	{
		if (abs(*iter1 - *iter2) > 5)
		{
			if (*iter1 < *iter2) iter1 += 2;
			else iter2 += 2;
		}
		else
		{
			rect1_regionIndex.push_back(*iter1);
			rect2_regionIndex.push_back(*iter2);
			rect1_regionIndex.push_back(*(++iter1));
			rect2_regionIndex.push_back(*(++iter2));
			++iter1;
			++iter2;
		}
	}

	std::cout << "rect1_regionIndex.size: " << rect1_regionIndex.size() << std::endl;
	std::cout << "rect2_regionIndex.size: " << rect2_regionIndex.size() << std::endl;
	for (int i = 0; i < rect1_regionIndex.size() / 2; i++)
	{
		std::cout << "region's index: " << i << std::endl;
		std::cout << "rect1_region: " << rect1_regionIndex[2 * i] << "; " << rect1_regionIndex[2 * i + 1] << std::endl;
		std::cout << "rect2_region: " << rect2_regionIndex[2 * i] << "; " << rect2_regionIndex[2 * i + 1] << std::endl;
	}

	// 利用相互匹配的子区域计算子区域的偏移量
	std::vector<int> croped_distance;
	for (int i = 0; i < rect1_regionIndex.size() / 2; i++)
	{
		// 这里要注意 rect2_region 的高小于 rect1_region 的高的情况
		cv::Mat rect1_region = rect1(cv::Rect(0, rect1_regionIndex[2 * i], 100,
			rect1_regionIndex[2 * i + 1] - rect1_regionIndex[2 * i]));
		cv::Mat rect2_region = rect2(cv::Rect(0, rect2_regionIndex[2 * i] - 1, 2 * 100,
			rect2_regionIndex[2 * i + 1] - rect2_regionIndex[2 * i] + 1));

		cv::Mat matchResult = cv::Mat(rect2_region.size(), rect2_region.type());
		cv::matchTemplate(rect2_region, rect1_region, matchResult, cv::TM_CCOEFF_NORMED);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

		croped_distance.push_back(rect2.cols - maxLoc.x);
	}

	// 部分子区域例如直线，在配准过程中容易出现错误，需要进行修正
	std::vector<int> croped_distance_revised;
	OffsetRevise(croped_distance, croped_distance_revised);
	for (int i = 0; i < croped_distance_revised.size(); i++)
	{
		std::cout << "croped_distance: " << croped_distance[i]
			<< "; croped_distance_revised: " << croped_distance_revised[i] << std::endl;
	}

	// 对间隔区域进行插值
	std::vector<int> croped_distance_revised_again(2 * croped_distance_revised.size() + 1);
	croped_distance_revised_again[0] = croped_distance_revised[0];
	croped_distance_revised_again.back() = croped_distance_revised.back();
	for (int i = 1; i < croped_distance_revised_again.size() - 1; i++)
	{
		if (i % 2 == 1)
		{
			croped_distance_revised_again[i] = croped_distance_revised[(i - 1) / 2];
		}
		else
		{
			int value = std::floor((croped_distance_revised[(i / 2) - 1] + croped_distance_revised[i / 2]) / 2);
			croped_distance_revised_again[i] = value;
		}
	}

	// 获取两个被切开区域的起始坐标和终止坐标
	cv::Mat dilate_used = matTmp.clone();
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
	cv::dilate(dilate_used, dilate_used, element);
	std::vector<int> dilated_used_regionIndex;
	GetVerProjRegions(dilate_used, dilated_used_regionIndex);
	int left_part_x = dilated_used_regionIndex[1];
	int right_part_x = dilated_used_regionIndex[dilated_used_regionIndex.size() - 2];

	// 对局部区域进行拼接，需要统计出最大偏移量和最小偏移量备用
	int offset_max = *max_element(croped_distance_revised_again.begin(), croped_distance_revised_again.end());
	int offset_min = *min_element(croped_distance_revised_again.begin(), croped_distance_revised_again.end());

	// 将首行和末尾一行插入rect1_region、rect2_region中，如此实现被切开区域的全覆盖
	rect1_regionIndex.insert(rect1_regionIndex.begin(), 0);
	rect1_regionIndex.insert(rect1_regionIndex.end(), matTmp.rows);
	rect2_regionIndex.insert(rect2_regionIndex.begin(), 0);
	rect2_regionIndex.insert(rect2_regionIndex.end(), matTmp.rows);

	std::cout << "rect1_regionIndex'size: " << rect1_regionIndex.size() << std::endl;
	std::cout << "rect2_regionIndex'size: " << rect2_regionIndex.size() << std::endl;
	std::cout << "croped_distance_revised_again's size: " << croped_distance_revised_again.size() << std::endl;

	std::vector<cv::Mat> region_merge;
	for (int i = 0; i < croped_distance_revised_again.size(); i++)
	{
		cv::Mat left_part = matTmp(cv::Rect(0, rect1_regionIndex[i],
			left_part_x + croped_distance_revised_again[i] - offset_min,
			rect1_regionIndex[i + 1] - rect1_regionIndex[i]));

		cv::Mat right_part = matTmp(cv::Rect(right_part_x, rect1_regionIndex[i],
			matTmp.cols-right_part_x-croped_distance_revised_again[i],
			rect1_regionIndex[i + 1] - rect1_regionIndex[i]));
		
		cv::Mat merge_temp;
		cv::hconcat(right_part, left_part, merge_temp);
		region_merge.push_back(merge_temp);
	}

	cv::Mat region_merged;
	cv::vconcat(region_merge, region_merged);
	std::cout << region_merged.size() << std::endl;

	cv::Mat region_surplus = matTmp(cv::Rect(left_part_x, 0, right_part_x - left_part_x, matTmp.rows));

	cv::Mat result;
	cv::hconcat(region_merged, region_surplus, result);

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