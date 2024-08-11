#include "Projection.h"

// ��ͶӰ�������ƽ��
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

//����ͼ�����ֱ����ͶӰ��������һ��ͼ��
cv::Mat GetVerProjImage(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	int height = matTmp.rows;
	int width = matTmp.cols;

	//����255�����Ŀ�����ڵ���
	int maxCol = 0, maxNum = 0;
	//����255��С��Ŀ�����ڵ���
	int minCol = 0, minNum = matTmp.rows;

	int tmp = 0;					//���浱ǰ�е�255��Ŀ
	int* projArray = new int[width];//����ÿһ��255��Ŀ������

	//ѭ������ͼ�����ݣ�����ÿһ�е�255�����Ŀ
	for (int col = 0; col < width; ++col)
	{
		tmp = 0;
		for (int row = 0; row < height; ++row)
		{
			// ��ɫ����
			if (matTmp.at<uchar>(row, col) > 100) ++tmp;
		}

		// ��¼��ǰ���а�ɫ���صĸ���
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

	// ���������ƴ�ֱͶӰͼ��
	cv::Mat projImg(height, width, CV_8UC1, cv::Scalar(255));

	for (int col = 0; col < width; ++col)
	{
		cv::line(projImg, cv::Point(col, height - projArray[col]),
			cv::Point(col, height - 1), cv::Scalar::all(0));
	}

	delete[] projArray;//ɾ��new����
	return  projImg;
}


//����ͼ�����ֱ����ͶӰ��������һ��ͼ��
int GetVerProjRegions(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	int height = matTmp.rows;
	int width = matTmp.cols;

	//����255�����Ŀ�����ڵ���
	int maxCol = 0, maxNum = 0;
	//����255��С��Ŀ�����ڵ���
	int minCol = 0, minNum = matTmp.rows;

	int tmp = 0;					//���浱ǰ�е�255��Ŀ
	int* projArray = new int[width];//����ÿһ��255��Ŀ������

	//ѭ������ͼ�����ݣ�����ÿһ�е�255�����Ŀ
	for (int col = 0; col < width; ++col)
	{
		tmp = 0;
		for (int row = 0; row < height; ++row)
		{
			// ��ɫ����
			if (matTmp.at<uchar>(row, col) > 100) ++tmp;
		}

		// ��¼��ǰ���а�ɫ���صĸ���
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
		// ����հ�����
		if (!inRegion && projArray[col] >= 3000)
		{
			inRegion = true;
			startIndex = col;
			regionIndex.push_back(startIndex);
		}

		// �����ǩ����
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

	delete[] projArray;//ɾ��new����
	return  nums;
}


//����ͼ���ˮƽ����ͶӰ��������һ��ͼ��
cv::Mat GetHorProjImage(const cv::Mat& image)
{
	cv::Mat matTmp = image.clone();
	int height = matTmp.rows;
	int width = matTmp.cols;

	//����255�����Ŀ�����ڵ���
	int maxCol = 0, maxNum = 0;
	//����255��С��Ŀ�����ڵ���
	int minCol = 0, minNum = matTmp.rows;

	int tmp = 0;						//���浱ǰ�е�255��Ŀ
	int* projArray = new int[height];	//����ÿһ��255��Ŀ������

	//ѭ������ͼ�����ݣ�����ÿһ�е�255�����Ŀ
	for (int row = 0; row < height; ++row)
	{
		tmp = 0;
		for (int col = 0; col < width; ++col)
		{
			// ��ɫ����
			if (matTmp.at<uchar>(row, col) > 100) ++tmp;
		}

		// ��¼��ǰ���а�ɫ���صĸ���
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
	//���������ƴ�ֱͶӰͼ��
	cv::Mat projImg(height, width, CV_8UC1, cv::Scalar(255));

	for (int row = 0; row < height; ++row)
	{
		cv::line(projImg, cv::Point(0, row),
			cv::Point(projArray[row], row), cv::Scalar::all(0));
	}

	delete[] projArray;//ɾ��new����
	return  projImg;
}

