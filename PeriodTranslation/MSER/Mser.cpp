#include "Mser.h"

cv::Mat MSERDetection(const cv::Mat& image)
{
	// MSER������£�GRAY��RGBͼ�����
	cv::Mat matTmp = image.clone();

	// ��ʼ�������
	cv::Ptr<cv::MSER> ptrMSER = cv::MSER::create();

	// �㼯������
	std::vector<std::vector<cv::Point>> points;

	// ���ε�����
	std::vector<cv::Rect> rects;

	// ��ʼ���
	ptrMSER->detectRegions(image, points, rects);

	// MSER������ʾ
	cv::Mat output(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));

	// ���ÿ����⵽�����������ڲ�ɫ������ʾ MSER
	// ������������ʾ�ϴ�� MSER
	cv::RNG rng;
	for (std::vector<std::vector<cv::Point> >::reverse_iterator
		it = points.rbegin(); it != points.rend(); ++it)
	{
		// ���������ɫ
		cv::Vec3b c(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));

		// ��� MSER �����е�ÿ����
		for (std::vector<cv::Point>::iterator itPts = it->begin();
			itPts != it->end(); ++itPts)
		{
			// ����д MSER ������
			if (output.at<cv::Vec3b>(*itPts)[0] == 255)
			{
				output.at<cv::Vec3b>(*itPts) = c;
			}
		}

		// ����MSER����������Բ
		cv::RotatedRect rect = cv::fitEllipse(*it);
		cv::ellipse(output, rect, cv::Scalar(255, 0, 0));
	}

	return output;
}
