#include "PreProcess.h"

cv::Mat MainProcess(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	int width = image.cols;
	int height = image.rows;

	cv::Mat dst;
	if (height >= width)
	{
		std::cout << "the image need to be rotated: " << std::endl;
		cv::rotate(matTmp, dst, cv::ROTATE_90_CLOCKWISE);
	}

	else
	{
		dst = matTmp.clone();
	}

	return dst;
}


cv::Mat BilateralFilter(const cv::Mat& image, int width, int sigmaSpace, int sigmaColor)
{
	cv::Mat matTmp = image.clone();
	cv::Mat dst = cv::Mat::zeros(matTmp.size(), matTmp.type());

	for (int i = width; i < matTmp.rows - width; i++)    //对每一个点进行处理
	{
		for (int j = width; j < matTmp.cols - width; j++)
		{
			double weightSum = 0;
			double filterValue = 0;
			for (int row_d = -(width / 2); row_d <= (width / 2); row_d++)   //以图像中的一点为中心，d为边长的方形区域内进行计算
			{
				for (int col_d = -(width / 2); col_d <= (width / 2); col_d++)
				{
					double distance_Square = row_d * row_d + col_d * col_d;
					double value_Square = pow((matTmp.at<uchar>(i, j) - matTmp.at<uchar>(i + row_d, j + col_d)), 2);
					double weight = exp(-1 * (distance_Square / (2 * sigmaSpace * sigmaSpace) + value_Square / (2 * sigmaColor * sigmaColor)));
					weightSum += weight;               //求滤波窗口内的权重和，用于归一化；
					filterValue += (weight * matTmp.at<uchar>(i + row_d, j + col_d));

				}
			}
			dst.at<uchar>(i, j) = filterValue / weightSum;
		}
	}
	return dst;
}
