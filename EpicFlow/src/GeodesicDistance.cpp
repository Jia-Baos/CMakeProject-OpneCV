#include "GeodesicDistance.hpp"

void CheckSeeds(const cv::Mat& dist, const std::vector<cv::Point2f> seeds)
{
	for (auto& iter : seeds)
	{
		const int x = iter.y;
		const int y = iter.x;

		if (dist.ptr<float>(x)[y] != 0.0)
		{
			std::cerr << "Error: distance of seed is larger than 0 !" << std::endl;
			exit(1);
		}
	}
}

void GetKernel(std::unordered_map<std::string, std::vector<int>>& kernel,
	std::vector<float>& squared_dist, bool backward)
{
	if (backward)
	{
		kernel.insert({ "h", {0, 1, 1, 1} });
		kernel.insert({ "w", {1, -1, 0, 1} });
		squared_dist = { 1.0,2.0,1.0,2.0 };
	}
	else
	{
		kernel.insert({ "h", {-1, -1, -1, 0} });
		kernel.insert({ "w", {-1, 0, 1, -1} });
		squared_dist = { 2.0, 1.0, 2.0, 1.0 };
	}
}

float GetDistance(const float alpha, const float p_val, const float q_val,
	const float scaling_factor, const float squared_dist,
	const DistType dist_type)
{
	float dist_pq = 0.0;
	if (dist_type == euclidean)
	{
		// calculate spatial distance
		// (part of gdm formula without intensity, so basically just euclidean distance)
		dist_pq = alpha * std::sqrt(squared_dist);
	}
	else if (dist_type == intensity)
	{
		// calculate intensity distance
		// (part of gdm formula without spatial distance, so basically absolute intensity difference)
		dist_pq = alpha * std::sqrt(std::pow((p_val - q_val), 2) * scaling_factor);
	}
	else
	{
		// calculate geodesic distance, combines spatial and intensity distance in image
		// in paper : dq = alpha * sqrt((G(p) - G(q)) ^ 2 + beta)
		dist_pq = alpha * std::sqrt(std::pow((p_val - q_val), 2) * scaling_factor + squared_dist);
	}

	return dist_pq;
}

void Pass2D(const cv::Mat& edges, cv::Mat& dist,
	std::unordered_map<std::string, std::vector<int>>& kernel,
	std::vector<float>& squared_dist, const float alpha,
	const float scaling_factor,
	const DistType dist_type,
	bool backward)
{
	const int width = edges.cols;
	const int height = edges.rows;

	// forward or backward loop
	const int base = backward ? -1 : 1;
	const int loopw1 = backward ? width - 1 : 0;
	const int looph1 = backward ? height - 1 : 0;
	const int loopw2 = backward ? -1 : width;
	const int looph2 = backward ? -1 : height;

	// pass through image
	for (size_t h = looph1; h != looph2; h += base)
	{
		for (size_t w = loopw1; w != loopw2; w += base)
		{
			// get distance and intensity value at point p
			float p_dist = dist.ptr<float>(h)[w];
			float p_val = edges.ptr<float>(h)[w];

			// looping through kernel
			for (size_t i = 0; i < kernel["h"].size(); i++)
			{
				const int nh = h + kernel["h"][i];
				const int nw = w + kernel["w"][i];

				// if selected pixel not in image continue
				if (nh < 0 || nh >= height || nw < 0 || nw >= width)
				{
					continue;
				}

				// distance and intensity value at point q of the kernel
				const float q_dist = dist.ptr<float>(nh)[nw];
				const float q_val = edges.ptr<float>(nh)[nw];
				const float squared_dist_pq = squared_dist[i];

				// compute distance between voxel p and q
				const float dist_pq = GetDistance(alpha, p_val, q_val,
					scaling_factor, squared_dist_pq, dist_type);

				// add distance between pixel p and pixel q to distance value in q
				// gives full distance to a seed point
				// Formula:
				// F * (p) = min(F(p), calc_distances) with calc_dist = 1 + dq + F * (q) for all places in kernel
				const float calc_dist = dist_pq + q_dist;

				// select minimal distance of calculated distance and current distance
					// for point p
				if (calc_dist < p_dist)
				{
					p_dist = calc_dist;
				}
			}

			// after going through kernel the selected distance is added to the distance map
			dist.ptr<float>(h)[w] = p_dist;
		}
	}
}

void GetDMRasterscan(const cv::Mat& edges, cv::Mat& dist,
	const std::vector<cv::Point2f>& seeds, const int its,
	const float scaling_factor,
	const DistType dist_type)
{
	// distance map raster scan
	// following paper from Toivanen et al
	// scaling_factor: change the weighting of euclidean distance vs intensity distance

	// edges is a 2D image with dimensions width, height
	// shape of image defines if the distance map will be 2D
	// seeds is a 2D array with dimensions #seeds by 2 (xy coords)
	// Image is not transposed, dimensions are width x height

	std::cout << "Compute distance map..." << std::endl;

	// setting all initial distances at a high value except the seeds..
	dist = cv::Mat(edges.size(), edges.type(), cv::Scalar(FLT_MAX));

	// ..the seeds are set to 0
	for (auto& iter : seeds)
	{
		const int x = iter.y;
		const int y = iter.x;

		dist.ptr<float>(x)[y] = 0.0;
	}

	for (size_t i = 0; i < its; i++)
	{
		// forward scan
		std::vector<float> squared_dist_f;
		std::unordered_map<std::string, std::vector<int>> kernel_f;

		GetKernel(kernel_f, squared_dist_f, false);
		Pass2D(edges, dist, kernel_f, squared_dist_f,
			1.0, scaling_factor, dist_type, false);

		// backward scan
		std::vector<float> squared_dist_b;
		std::unordered_map<std::string, std::vector<int>> kernel_b;

		GetKernel(kernel_b, squared_dist_b, true);
		Pass2D(edges, dist, kernel_b, squared_dist_b,
			1.0, scaling_factor, dist_type, true);
	}

	// sanity check :
	// check if the distance in any of the seeds is not 0 anymore
	// would mean something is wrong
	CheckSeeds(dist, seeds);
}
