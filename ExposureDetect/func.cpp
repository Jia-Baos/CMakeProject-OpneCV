#include "./func.hpp"

/**********************辅助函数**********************/

std::string ReplaceAll(const std::string& src, const std::string& old_value, const std::string& new_value)
{
    std::string dst = src;
    // 每次重新定位起始位置，防止上轮替换后的字符串形成新的old_value
    for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length())
    {
        if ((pos = dst.find(old_value, pos)) != std::string::npos)
        {
            dst.replace(pos, old_value.length(), new_value);
        }
        else { break; }
    }
    return dst;
}

void BorderTest(int& roi_x1, int& roi_y1, int& roi_x2, int& roi_y2, const int w, const int h)
{
    // std::cout << "Border Test..." << std::endl;
    roi_x1 = roi_x1 < 0 ? 0 : roi_x1;
    roi_y1 = roi_y1 < 0 ? 0 : roi_y1;
    roi_x2 = roi_x2 > w ? w : roi_x2;
    roi_y2 = roi_y2 > h ? h : roi_y2;
}

float valWeighted(const cv::Mat& src) {
    assert(src.type() == CV_8UC1);

    const int w = src.cols;
    const int h = src.rows;

    float array[256] = { 0.0f };
    // Compute the nums of pixels of every channels
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            int val = src.ptr<uchar>(i)[j];
            array[val]++;
        }
    }

    // Compute the probability of every gray value and entropy
    float val_sta = 0.0f;
    float* array_ptr = array;
    for (int i = 0; i < 255; i++)
    {
        float val_prob = *array_ptr / (w * h);
        val_sta += i * val_prob;
        array_ptr++;
    }

    return val_sta;
}

float ColorDisSta(const cv::Mat& src) {
    assert(src.type() == CV_8UC3);

    const int w = src.cols;
    const int h = src.rows;
    float array_b[256] = { 0.0f };
    float array_g[256] = { 0.0f };
    float array_r[256] = { 0.0f };
    // Compute the nums of pixels of every channels
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            int val_b = src.at<cv::Vec3b>(i, j)[0];
            int val_g = src.at<cv::Vec3b>(i, j)[1];
            int val_r = src.at<cv::Vec3b>(i, j)[2];

            array_b[val_b]++;
            array_g[val_g]++;
            array_r[val_r]++;
        }
    }

    // Compute the probability of every gray value and entropy
    float val_b_sta = 0.0f;
    float val_g_sta = 0.0f;
    float val_r_sta = 0.0f;
    float* array_b_ptr = array_b;
    float* array_g_ptr = array_g;
    float* array_r_ptr = array_r;

    for (int i = 0; i < 255; i++)
    {
        float val_b_prob = *array_b_ptr / (w * h);
        float val_g_prob = *array_g_ptr / (w * h);
        float val_r_prob = *array_r_ptr / (w * h);

        val_b_sta += i * val_b_prob;
        val_g_sta += i * val_g_prob;
        val_r_sta += i * val_r_prob;

        array_b_ptr++;
        array_g_ptr++;
        array_r_ptr++;
    }

    std::vector<float> vec{ val_b_sta, val_g_sta, val_r_sta };
    std::sort(vec.begin(), vec.end());

    // 对距离进行归一化，为了避免最小值、最大值的影响，直接提取中值
    // 此处最终还是提取了最小值，实践中发现中值存在错误提取地板的情况
    float max_dis = (255.0f - vec[0]) / 255.0f;

    return max_dis;
}

float Entropy(const cv::Mat& src)
{
    assert(src.type() == CV_8UC1);

    const int w = src.cols;
    const int h = src.rows;

    float array_gray[256] = { 0.0f };
    // Compute the nums of every gray value
    for (int i = 0; i < h; i++)
    {
        const uchar* src_ptr = src.ptr<uchar>(i);
        for (int j = 0; j < w; j++)
        {
            int gray_value = src_ptr[j];
            array_gray[gray_value]++;
        }
    }

    // Compute the probability of every gray value and entropy
    float my_entropy = 0.0f;
    float* array_gray_ptr = array_gray;

    for (int i = 0; i < 255; i++)
    {
        float gray_value_prob = *array_gray_ptr / (w * h);
        if (std::fabs(gray_value_prob) > 1e-6)
        {
            // 对熵值进行归一化
            // Sum(-Pi*log(Pi)) < Sum(-(1/(w*h))*log(1/(w*h))) -> Sum(-Pi*log(Pi)) < log(-1/(w*h))
            float curr_entropy = -gray_value_prob * std::log(gray_value_prob) / std::log(w * h);
            my_entropy += curr_entropy;
        }
        array_gray_ptr++;
    }

    return my_entropy;
}

/**********************关键函数**********************/
void getSeeds(std::vector<cv::Point2f>& seeds, const int w, const int h, const int step)
{
    const int gridw = w / step;
    const int gridh = h / step;
    const int ofsx = (w - (gridw - 1) * step) / 2;
    const int ofsy = (h - (gridh - 1) * step) / 2;
    const int nseeds = gridw * gridh;
    seeds.resize(nseeds);

    for (int i = 0; i < nseeds; i++)
    {
        const int x = i % gridw;
        const int y = i / gridw;

        // 保证 seed 不会出现在图像边缘上
        const float seedx = static_cast<float>(x * step + ofsx);
        const float seedy = static_cast<float>(y * step + ofsy);
        seeds[i] = cv::Vec2f(seedx, seedy);
    }
}

void getDescs(const cv::Mat& src, cv::Mat& score,
    const std::vector<cv::Point2f>& seeds, const int step, const int width)
{
    assert(src.type() == CV_8UC3);
    const int w = src.cols;
    const int h = src.rows;
    const int gridw = w / step;
    const int gridh = h / step;
    const int nseeds = seeds.size();

    // 初始化容器大小
    score = cv::Mat(gridh, gridw, CV_32FC1);

    //// ForDebug
    //cv::Mat value = cv::Mat(gridh, gridw, CV_32FC1);
    //cv::Mat color_dis = cv::Mat(gridh, gridw, CV_32FC1);
    //cv::Mat entropy = cv::Mat(gridh, gridw, CV_32FC1);

    // 获取亮度通道信息，为了提取过曝区域（白色）因此将颜色空间从HSV更改为HLS
    // 参考连接：https://blog.csdn.net/ahelloyou/article/details/111238932
    cv::Mat src_split{};
    std::vector<cv::Mat> vec{};
    cv::cvtColor(src, src_split, cv::COLOR_BGR2HSV);
    cv::split(src_split, vec);
    cv::Mat src_value = vec.back();

    //cv::Mat src_split{};
    //std::vector<cv::Mat> vec{};
    //cv::cvtColor(src, src_split, cv::COLOR_BGR2HLS);
    //cv::split(src_split, vec);
    //cv::Mat src_value = vec.at(1);

    // 获取灰度图信息
    cv::Mat src_gray{};
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

#ifdef OMP
#pragma omp parallel for num_threads(max_threads)
#endif // !OMP
    for (int i = 0; i < nseeds; i++)
    {
        const int center_x = seeds[i].x;
        const int center_y = seeds[i].y;
        int roi_x1 = center_x - width;
        int roi_y1 = center_y - width;
        int roi_x2 = center_x + width + 1;
        int roi_y2 = center_y + width + 1;

        // 防止patch超出区域
        BorderTest(roi_x1, roi_y1, roi_x2, roi_y2, w, h);

        const cv::Mat roi_value = src_value(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
        const cv::Mat roi_color = src(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
        //const cv::Mat roi_entropy = src_gray(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));

        //value.ptr<float>(i / gridw)[i % gridw] = cv::mean(roi_value)[0];
        //color_dis.ptr<float>(i / gridw)[i % gridw] = std::exp(-ColorDisSta(roi_color));
        //entropy.ptr<float>(i / gridw)[i % gridw] = Entropy(roi_entropy);

        //score.ptr<float>(i / gridw)[i % gridw] = valWeighted(roi_value) * std::exp(-ColorDisSta(roi_color));
        //score.ptr<float>(i / gridw)[i % gridw] = cv::mean(roi_value)[0] * std::exp(-ColorDisSta(roi_color));
        score.ptr<float>(i / gridw)[i % gridw] = cv::mean(roi_value)[0];
    }
}

void recoverScore(const std::vector<cv::Point2f>& seeds, cv::Mat& score,
    const cv::Size& image_size, const int step)
{
    assert(score.type() == CV_32FC1);

    const int radius = step;
    const int w = image_size.width;
    const int h = image_size.height;
    const int nmatches = seeds.size();

    cv::Mat score_norm = cv::Mat::zeros(h, w, CV_32FC1);
    for (int i = 0; i < nmatches; i++)
    {
        const int x = i % score.cols;
        const int y = i / score.cols;
        const float val = score.ptr<float>(y)[x];

        // draw each match as a radius*radius color block, 图像边缘部分并未被填充，
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                const int x = std::max(0, std::min(static_cast<int>(seeds[i].x + dx + 0.5f), w - 1));
                const int y = std::max(0, std::min(static_cast<int>(seeds[i].y + dy + 0.5f), h - 1));
                score_norm.ptr<float>(y)[x] = val;
            }
        }
    }
    score = score_norm.clone();
}

void segmentMap(const cv::Mat& src, const cv::Mat& templ, cv::Mat& dst, const float tolerance) {
    assert(src.type() == CV_32FC1);

    const int w = src.cols;
    const int h = src.rows;
    dst = cv::Mat::zeros(h, w, CV_32FC1);
    for (size_t i = 0; i < h; ++i) {
        for (size_t j = 0; j < w; ++j) {
            float val = src.ptr<float>(i)[j] - templ.ptr<float>(i)[j];
            if (val > tolerance) {
                dst.ptr<float>(i)[j] = static_cast<float>(pixel_type::overexposed);
            }
            else if (val > -tolerance) {
                dst.ptr<float>(i)[j] = static_cast<float>(pixel_type::normal);
            }
            else {
                dst.ptr<float>(i)[j] = static_cast<float>(pixel_type::underexposed);
            }
        }
    }
}

void segmentMap(const cv::Mat& src, cv::Mat& dst, const float thresh_low, const float thresh_high) {
    assert(src.type() == CV_32FC1);

    const int w = src.cols;
    const int h = src.rows;
    dst = cv::Mat(h, w, CV_8UC1, cv::Scalar::all(static_cast<uchar>(pixel_type::normal)));

    for (size_t i = 0; i < h; ++i) {
        for (size_t j = 0; j < w; ++j) {
            float val = src.ptr<float>(i)[j];
            if (val > thresh_high) {
                dst.ptr<uchar>(i)[j] = static_cast<uchar>(pixel_type::overexposed);
            }
            else if (val < thresh_low) {
                dst.ptr<uchar>(i)[j] = static_cast<uchar>(pixel_type::underexposed);
            }
        }
    }
}

void fillHoles(const cv::Mat& src, cv::Mat& dst, const int erode_nums, const int dilate_nums) {
    dst = src.clone();

    // 因为只想获取过曝区域的信息，所以把曝光不足、正常区域屏蔽
    dst.setTo(0, dst < 9.0f);

    // 为了保证空洞在腐蚀后被稳定填充，故膨胀操作迭代次数应略大于腐蚀操作迭代次数
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::erode(dst, dst, kernel, cv::Point(-1, -1), erode_nums);
    cv::dilate(dst, dst, kernel, cv::Point(-1, -1), dilate_nums);
}

void fillHoles(const cv::Mat& src, cv::Mat& dst, const int erode_nums) {
    dst = src.clone();

    // 因为只想获取过曝区域的信息，所以把曝光不足、正常区域屏蔽
    dst.setTo(0, dst < 9.0f);

    // 为了保证空洞在腐蚀后被稳定填充，故膨胀操作迭代次数应略大于腐蚀操作迭代次数
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::erode(dst, dst, kernel, cv::Point(-1, -1), erode_nums);
    cv::dilate(dst, dst, kernel, cv::Point(-1, -1), erode_nums + 3);
}

void connectedDomainAnalysis(const cv::Mat& src, cv::Mat& dst) {
    const int w = src.cols;
    const int h = src.rows;
    cv::RNG rng(1025);  // 生成随机颜色，用于区分不同连通域
    cv::Mat out;
    cv::Mat status, centroids;
    int numberOfRegions = cv::connectedComponentsWithStats(src, out, status, centroids, 8, CV_16U); // 统计图像连通域的个数

    std::vector<cv::Vec3b> colors;  // 用于储存每个连通区域的颜色（三通道）
    std::vector<cv::Point2f> points;    // 用于储存每个连通区域的质心位置
    for (int i = 0; i < numberOfRegions; ++i)
    {
        cv::Vec3b vec3 = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));  // 给定随机色
        colors.push_back(vec3);
        points.push_back(cv::Point2f(centroids.ptr<double>(i)[0], centroids.ptr<double>(i)[1]));
    }

    // 以不同颜色标记出不同的连通域
    cv::Mat colorMap = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int row = 0; row < h; ++row)
    {
        for (int col = 0; col < w; ++col)
        {
            int label = out.ptr<uint16_t>(row)[col];
            if (label == 0) // 背景的黑色不改变
            {
                continue;
            }
            colorMap.ptr<cv::Vec3b>(row)[col] = colors[label];
        }
    }
    dst = colorMap.clone();
}
