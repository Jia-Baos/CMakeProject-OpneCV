#pragma

#include <algorithm>
#include <omp.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#define OMP
//#define DEBUG
const int max_threads = 4;

struct params_t {
    int step = 3;
    int width = 2;
    int thresh_low = 20;
    int thresh_high = 200;
    int erode_num = 3;
    int dilate_num = 6;
};

enum class pixel_type
{
    underexposed = 3,
    normal = 6,
    overexposed = 9
};

std::string ReplaceAll(const std::string& src, const std::string& old_value,
    const std::string& new_value);

void getSeeds(std::vector<cv::Point2f>& seeds, const int w, const int h, const int step = 5);

void getDescs(const cv::Mat& image, cv::Mat& score, const std::vector<cv::Point2f>& seeds, const int step, const int width);

void recoverScore(const std::vector<cv::Point2f>& seeds, cv::Mat& score, const cv::Size& image_size, const int step);

void segmentMap(const cv::Mat& src, const cv::Mat& templ, cv::Mat& dst, const float tolerance);

void segmentMap(const cv::Mat& src, cv::Mat& dst, const float thresh_low, const float thresh_high);

void fillHoles(const cv::Mat& src, cv::Mat& dst, const int erode_nums, const int dilate_nums);

void fillHoles(const cv::Mat& src, cv::Mat& dst, const int erode_nums);

void connectedDomainAnalysis(const cv::Mat& src, cv::Mat& dst);
