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

	//随机种子
	std::srand(std::time(0));
	float datacost = 0;
	float datacost_best = FLT_MAX;

	// 设置循环的次数
	while (--max_iter_)
	{
		std::cout << "iter: " << max_iter_ << std::endl;

		std::vector<int> maybeInliers;		// 从数据中随机选择n个拟合的数据组
		std::vector<int> alsoInliers;		// 根据当前模型参数从数据中筛选内点
		Plane maybeModel;					// 根据以上n个数据计算得到的模型参数

		// 随机选取三个点，并计算其形成的平面
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
			// 计算maybeInliers外的其他数据在当前模型参数下的代价，这里重复计算了已有的内点
			// 点到平面的距离公式
			float dis =
				fabs(maybeModel.a * iter->x + maybeModel.b * iter->y + maybeModel.c * iter->z + maybeModel.d)
				/ std::sqrt(std::pow(maybeModel.a, 2) + std::pow(maybeModel.b, 2) + std::pow(maybeModel.c, 2));
			if (dis < threshold_)
			{
				alsoInliers.push_back(iter - points_3d_.begin());
				datacost = datacost + dis;
			}
		}

		// 如果当前模型参数下代价更小，则更新模型参数
		if (datacost < datacost_best)
		{
			datacost_best = datacost;
			this->plane_best_.a = maybeModel.a;
			this->plane_best_.b = maybeModel.b;
			this->plane_best_.c = maybeModel.c;
			this->plane_best_.d = maybeModel.d;
		}

		// 输出当前平面参数
		std::cout << "Current Model's Parameters: "
			<< datacost << " "
			<< maybeModel.a << " "
			<< maybeModel.b << " "
			<< maybeModel.c << " "
			<< maybeModel.d << std::endl;
	}

	// 输出最优平面参数
	std::cout << "Best Model's Parameters: "
		<< this->plane_best_.a << " "
		<< this->plane_best_.b << " "
		<< this->plane_best_.c << " "
		<< this->plane_best_.d << std::endl;
}