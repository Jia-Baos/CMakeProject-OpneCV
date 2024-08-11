#include "RANSAC.h"

RANSAC::RANSAC()
{
	max_iter_ = 20;
	threshold_ = 1;
	std::cout << "Initalize the Class: RANSAC" << std::endl;
}


RANSAC::~RANSAC()
{
	std::cout << "Destory the Class: RANSAC" << std::endl;
}


void RANSAC::SetInput(const std::vector<cv::Point3f>& points_3d)
{
	this->points_3d_.assign(points_3d.begin(), points_3d.end());
}


void RANSAC::Compute(const std::vector<cv::Point3f>& points_3d)
{
	this->SetInput(points_3d);

	//�������
	std::srand(std::time(0));
	float datacost = 0;
	float datacost_best = FLT_MAX;

	// ����ѭ���Ĵ���
	while (--max_iter_)
	{
		std::cout << "iter: " << max_iter_ << std::endl;

		std::vector<int> maybeInliers;		// �����������ѡ��n����ϵ�������
		std::vector<int> alsoInliers;		// ���ݵ�ǰģ�Ͳ�����������ɸѡ�ڵ�
		Plane maybeModel;					// ��������n�����ݼ���õ���ģ�Ͳ���

		// ���ѡȡ�����㣬���������γɵ�ƽ��
		while (maybeInliers.size() < 3)
		{
			int index = std::rand() % points_3d_.size();
			if (std::find(maybeInliers.begin(), maybeInliers.end(), index) == maybeInliers.end())
			{
				maybeInliers.push_back(index);
			}
		}

		auto idx = maybeInliers.begin();
		float x1 = points_3d_.at(*idx).x, y1 = points_3d_.at(*idx).y, z1 = points_3d_.at(*idx).z;
		++idx;
		float x2 = points_3d_.at(*idx).x, y2 = points_3d_.at(*idx).y, z2 = points_3d_.at(*idx).z;
		++idx;
		float x3 = points_3d_.at(*idx).x, y3 = points_3d_.at(*idx).y, z3 = points_3d_.at(*idx).z;

		maybeModel.a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1);
		maybeModel.b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1);
		maybeModel.c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
		maybeModel.d = -(maybeModel.a * x1 + maybeModel.b * y1 + maybeModel.c * z1);

		for (auto iter = points_3d_.begin(); iter != points_3d_.end(); ++iter)
		{
			// ����maybeInliers������������ڵ�ǰģ�Ͳ����µĴ��ۣ������ظ����������е��ڵ�
			// �㵽ƽ��ľ��빫ʽ
			float dis =
				fabs(maybeModel.a * iter->x + maybeModel.b * iter->y + maybeModel.c * iter->z + maybeModel.d)
				/ std::sqrt(std::pow(maybeModel.a, 2) + std::pow(maybeModel.b, 2) + std::pow(maybeModel.c, 2));
			if (dis < threshold_)
			{
				alsoInliers.push_back(iter - points_3d_.begin());
				datacost = datacost + dis;
			}
		}

		// �����ǰģ�Ͳ����´��۸�С�������ģ�Ͳ���
		if (datacost < datacost_best)
		{
			datacost_best = datacost;
			this->plane_best_.a = maybeModel.a;
			this->plane_best_.b = maybeModel.b;
			this->plane_best_.c = maybeModel.c;
			this->plane_best_.d = maybeModel.d;
		}

		// �����ǰƽ�����
		std::cout << "Current Model's Parameters: "
			<< datacost << " "
			<< maybeModel.a << " "
			<< maybeModel.b << " "
			<< maybeModel.c << " "
			<< maybeModel.d << std::endl;
	}

	// �������ƽ�����
	std::cout << "Best Model's Parameters: "
		<< this->plane_best_.a << " "
		<< this->plane_best_.b << " "
		<< this->plane_best_.c << " "
		<< this->plane_best_.d << std::endl;
}