#include "Projection.h"

// 对投影结果进行平滑
void GaussSmoothOriHist(int* hist, int n)
{
	int prev = hist[n - 1], temp, h0 = hist[0];

	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.3 * prev + 0.4 * hist[i] +
			0.3 * (i + 1 >= n ? h0 : hist[i + 1]);
		prev = temp;
	}

}

//计算图像的竖直方向投影，并返回一幅图像
cv::Mat GetVerProjImage(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	int height = matTmp.rows;
	int width = matTmp.cols;

	//重置255最大数目和所在的列
	int maxCol = 0, maxNum = 0;
	//重置255最小数目和所在的列
	int minCol = 0, minNum = matTmp.rows;

	int tmp = 0;					//保存当前行的255数目
	int* projArray = new int[width];//保存每一行255数目的数组

	//循环访问图像数据，查找每一行的255点的数目
	for (int col = 0; col < width; ++col)
	{
		tmp = 0;
		for (int row = 0; row < height; ++row)
		{
			// 白色像素
			if (matTmp.at<uchar>(row, col) > 100) ++tmp;
		}

		// 记录当前列中白色像素的个数
		projArray[col] = tmp;

		if (tmp > maxNum)
		{
			maxNum = tmp;
			maxCol = col;
		}
		if (tmp < minNum)
		{
			minNum = tmp;
			minCol = col;
		}
	}

	// 创建并绘制垂直投影图像
	cv::Mat projImg(height, width, CV_8UC1, cv::Scalar(255));

	for (int col = 0; col < width; ++col)
	{
		cv::line(projImg, cv::Point(col, height - projArray[col]),
			cv::Point(col, height - 1), cv::Scalar::all(0));
	}

	delete[] projArray;//删除new数组
	return  projImg;
}


//计算图像的竖直方向投影，并返回一幅图像
int GetVerProjRegions(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	int height = matTmp.rows;
	int width = matTmp.cols;

	//重置255最大数目和所在的列
	int maxCol = 0, maxNum = 0;
	//重置255最小数目和所在的列
	int minCol = 0, minNum = matTmp.rows;

	int tmp = 0;					//保存当前行的255数目
	int* projArray = new int[width];//保存每一行255数目的数组

	//循环访问图像数据，查找每一行的255点的数目
	for (int col = 0; col < width; ++col)
	{
		tmp = 0;
		for (int row = 0; row < height; ++row)
		{
			// 白色像素
			if (matTmp.at<uchar>(row, col) > 100) ++tmp;
		}

		// 记录当前列中白色像素的个数
		projArray[col] = tmp;

		if (tmp > maxNum)
		{
			maxNum = tmp;
			maxCol = col;
		}
		if (tmp < minNum)
		{
			minNum = tmp;
			minCol = col;
		}
	}

	int startIndex = 0;
	int endIndex = 0;
	bool inRegion = false;
	std::vector<int> regionIndex;

	for (int col = 0; col < width; ++col)
	{
		// 进入空白区域
		if (!inRegion && projArray[col] >= 3000)
		{
			inRegion = true;
			startIndex = col;
			regionIndex.push_back(startIndex);
		}

		// 进入标签区域
		else if (inRegion && projArray[col] < 3000)
		{
			inRegion = false;
			endIndex = col;
			regionIndex.push_back(endIndex);
		}
	}

	int nums = 0;
	for (int i = 0; i < regionIndex.size() / 2; i++)
	{
		startIndex = regionIndex[i];
		endIndex = regionIndex[i + 1];
		if ((endIndex - startIndex) > 200)
		{
			nums++;
		}
		else {}
	}

	delete[] projArray;//删除new数组
	return  nums;
}


//计算图像的水平方向投影，并返回一幅图像
cv::Mat GetHorProjImage(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	int height = matTmp.rows;
	int width = matTmp.cols;

	//重置255最大数目和所在的行
	int maxCol = 0, maxNum = 0;
	//重置255最小数目和所在的行
	int minCol = 0, minNum = matTmp.rows;

	int tmp = 0;						//保存当前行的255数目
	int* projArray = new int[height];	//保存每一行255数目的数组

	//循环访问图像数据，查找每一行的255点的数目
	for (int row = 0; row < height; ++row)
	{
		tmp = 0;
		for (int col = 0; col < width; ++col)
		{
			// 白色像素
			if (matTmp.at<uchar>(row, col) > 100) ++tmp;
		}

		// 记录当前列中白色像素的个数
		projArray[row] = tmp;

		if (tmp > maxNum)
		{
			maxNum = tmp;
			maxCol = row;
		}
		if (tmp < minNum)
		{
			minNum = tmp;
			minCol = row;
		}
	}
	//创建并绘制垂直投影图像
	cv::Mat projImg(height, width, CV_8UC1, cv::Scalar(255));

	for (int row = 0; row < height; ++row)
	{
		cv::line(projImg, cv::Point(0, row),
			cv::Point(projArray[row], row), cv::Scalar::all(0));
	}

	delete[] projArray;//删除new数组
	return  projImg;
}

