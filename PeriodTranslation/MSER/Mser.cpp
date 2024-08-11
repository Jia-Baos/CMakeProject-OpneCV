#include "Mser.h"

cv::Mat MSERDetection(const cv::Mat& image)
{
	// MSER检测器下：GRAY、RGB图像均可
	cv::Mat matTmp = image.clone();

	// 初始化检测器
	cv::Ptr<cv::MSER> ptrMSER = cv::MSER::create();

	// 点集的容器
	std::vector<std::vector<cv::Point>> points;

	// 矩形的容器
	std::vector<cv::Rect> rects;

	// 开始检测
	ptrMSER->detectRegions(image, points, rects);

	// MSER区域显示
	cv::Mat output(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));

	// 针对每个检测到的特征区域，在彩色区域显示 MSER
	// 反向排序，先显示较大的 MSER
	cv::RNG rng;
	for (std::vector<std::vector<cv::Point> >::reverse_iterator
		it = points.rbegin(); it != points.rend(); ++it)
	{
		// 生成随机颜色
		cv::Vec3b c(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));

		// 针对 MSER 集合中的每个点
		for (std::vector<cv::Point>::iterator itPts = it->begin();
			itPts != it->end(); ++itPts)
		{
			// 不重写 MSER 的像素
			if (output.at<cv::Vec3b>(*itPts)[0] == 255)
			{
				output.at<cv::Vec3b>(*itPts) = c;
			}
		}

		// 绘制MSER区域的拟合椭圆
		cv::RotatedRect rect = cv::fitEllipse(*it);
		cv::ellipse(output, rect, cv::Scalar(255, 0, 0));
	}

	return output;
}
