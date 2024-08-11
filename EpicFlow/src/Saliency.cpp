#include "Saliency.hpp"

void GaussianSmoothing(cv::Mat& src, const float sigma)
{
	// 向上取整
	int radius = std::ceil(3 * sigma);
	// 不论radius为奇还是偶，ksize始终为奇
	int ksize = 2 * radius + 1;

	cv::GaussianBlur(src, src, cv::Size(ksize, ksize), sigma);
}

void GaussianSmoothing(const cv::Mat& src, cv::Mat& dst, const float sigma)
{
	// 向上取整
	int radius = std::ceil(3 * sigma);
	// 不论radius为奇还是偶，ksize始终为奇
	int ksize = 2 * radius + 1;

	cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), sigma);
}

void ComputeGradient(const cv::Mat& src, cv::Mat& Fx, cv::Mat& Fy)
{
	cv::Mat src_pad;
	cv::copyMakeBorder(src, src_pad,
		1, 1, 1, 1, cv::BORDER_CONSTANT);

	Fx = cv::Mat::zeros(src.size(), src.type());
	Fy = cv::Mat::zeros(src.size(), src.type());

	// Single or multiple channels
	if (src.channels() == 1)
	{
		for (int i = 0; i < src.rows; i++)
		{
			float* ptr_Fx = Fx.ptr<float>(i);
			float* ptr_Fy = Fy.ptr<float>(i);

			for (int j = 0; j < src.cols; j++)
			{
				// 水平方向的梯度
				ptr_Fx[j] = (src_pad.ptr<float>(i + 1)[j + 2]
					- src_pad.ptr<float>(i + 1)[j]) / 2.0;
				// 竖直方向的梯度
				ptr_Fy[j] = (src_pad.ptr<float>(i + 2)[j + 1]
					- src_pad.ptr<float>(i)[j + 1]) / 2.0;
			}
		}
	}
	else if (src.channels() == 3)
	{
		for (int i = 0; i < src.rows; i++)
		{
			cv::Vec3f* ptr_Fx = Fx.ptr<cv::Vec3f>(i);
			cv::Vec3f* ptr_Fy = Fy.ptr<cv::Vec3f>(i);

			for (int j = 0; j < src.cols; j++)
			{
				// 水平方向的梯度
				ptr_Fx[j][0] = (src_pad.ptr<cv::Vec3f>(i + 1)[j + 2][0]
					- src_pad.ptr<cv::Vec3f>(i + 1)[j][0]) / 2.0;
				ptr_Fx[j][1] = (src_pad.ptr<cv::Vec3f>(i + 1)[j + 2][1]
					- src_pad.ptr<cv::Vec3f >(i + 1)[j][1]) / 2.0;
				ptr_Fx[j][2] = (src_pad.ptr<cv::Vec3f>(i + 1)[j + 2][2]
					- src_pad.ptr<cv::Vec3f>(i + 1)[j][2]) / 2.0;

				// 竖直方向的梯度
				ptr_Fy[j][0] = (src_pad.ptr<cv::Vec3f>(i + 2)[j + 1][0]
					- src_pad.ptr<cv::Vec3f>(i)[j + 1][0]) / 2.0;
				ptr_Fy[j][1] = (src_pad.ptr<cv::Vec3f>(i + 2)[j + 1][1]
					- src_pad.ptr<cv::Vec3f>(i)[j + 1][1]) / 2.0;
				ptr_Fy[j][2] = (src_pad.ptr<cv::Vec3f>(i + 2)[j + 1][2]
					- src_pad.ptr<cv::Vec3f>(i)[j + 1][2]) / 2.0;
			}
		}
	}
}

void Saliency(const cv::Mat& src, cv::Mat& dst,
	float sigma_image, float sigma_matrix)
{
	// Smooth the image
	cv::Mat sim = cv::Mat::zeros(src.size(), src.type());
	GaussianSmoothing(src, sim, sigma_image);

	// Compute the derivatives
	cv::Mat imx = cv::Mat::zeros(src.size(), src.type());
	cv::Mat imy = cv::Mat::zeros(src.size(), src.type());
	ComputeGradient(sim, imx, imy);

	// Compute autocorrelation matrix
	cv::Mat imxx = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat imxy = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat imyy = cv::Mat::zeros(src.size(), CV_32FC1);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			imxx.ptr<float>(i)[j] =
				imx.ptr<cv::Vec3f>(i)[j][0] * imx.ptr<cv::Vec3f>(i)[j][0]
				+ imx.ptr<cv::Vec3f>(i)[j][1] * imx.ptr<cv::Vec3f>(i)[j][1]
				+ imx.ptr<cv::Vec3f>(i)[j][2] * imx.ptr<cv::Vec3f>(i)[j][2];

			imxy.ptr<float>(i)[j] =
				imx.ptr<cv::Vec3f>(i)[j][0] * imy.ptr<cv::Vec3f>(i)[j][0]
				+ imx.ptr<cv::Vec3f>(i)[j][1] * imy.ptr<cv::Vec3f>(i)[j][1]
				+ imx.ptr<cv::Vec3f>(i)[j][2] * imy.ptr<cv::Vec3f>(i)[j][2];

			imyy.ptr<float>(i)[j] =
				imy.ptr<cv::Vec3f>(i)[j][0] * imy.ptr<cv::Vec3f>(i)[j][0]
				+ imy.ptr<cv::Vec3f>(i)[j][1] * imy.ptr<cv::Vec3f>(i)[j][1]
				+ imy.ptr<cv::Vec3f>(i)[j][2] * imy.ptr<cv::Vec3f>(i)[j][2];
		}
	}

	// Integrate autocorrelation matrix
	GaussianSmoothing(imxx, sigma_matrix);
	GaussianSmoothing(imxy, sigma_matrix);
	GaussianSmoothing(imyy, sigma_matrix);

	// Compute smallest eigenvalue
	float zeros = 0.0f;
	float half = 0.5;
	dst = cv::Mat::zeros(src.size(), CV_32FC1);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			float tmp = half * (imxx.ptr<float>(i)[j] + imyy.ptr<float>(i)[j]);
			tmp = std::sqrt(std::max(zeros, tmp -
				std::sqrt(std::max(zeros, tmp * tmp
					+ imxy.ptr<float>(i)[j] * imxy.ptr<float>(i)[j]
					- imxx.ptr<float>(i)[j] * imyy.ptr<float>(i)[j]))));
			dst.ptr<float>(i)[j] = tmp;
		}
	}
}
