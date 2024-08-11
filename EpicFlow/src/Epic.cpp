#include "./Epic.hpp"

/// @brief create a copy of input matches with 4 columns, with all points inside the image area
std::vector<cv::Vec4f> rectify_corres(const std::vector<cv::Vec4f>& matches,
    const int w, const int h, const int n_thread)
{
    const int matches_num = matches.size();
    std::vector<cv::Vec4f> res(matches_num, cv::Vec4f(0.0, 0.0, 0.0, 0.0));

#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(n_thread)
#endif
    for (size_t i = 0; i < matches_num; ++i)
    {
        res[i][0] = MY_MAX(0.0f, MY_MIN(matches[i][0], w - 1.0f));
        res[i][1] = MY_MAX(0.0f, MY_MIN(matches[i][1], h - 1.0f));
        res[i][2] = MY_MAX(0.0f, MY_MIN(matches[i][2], w - 1.0f));
        res[i][3] = MY_MAX(0.0f, MY_MIN(matches[i][3], h - 1.0f));
    }

    return res;
}

void epic(cv::Mat& flowx, cv::Mat& flowy, const cv::Mat& im,
    const std::vector<cv::Vec4f>& input_matches, cv::Mat& edges,
    const epic_params_t& params, const int n_thread)
{
    // copy matches and correct them if necessary
    std::vector<cv::Vec4f> matches = rectify_corres(input_matches, im.cols, im.rows, n_thread);
    if (params.verbose) {
        std::cout << "input matches size: " << matches.size() << std::endl;
    }

    std::cout << edges.ptr<float>(5)[5] << std::endl;
    // eventually add a constant to edges cost
    if (params.euc) { edges += params.euc; }
    std::cout << edges.ptr<float>(5)[5] << std::endl;

    // saliency filter
    if (params.saliency_th)
    {
        cv::Mat image_saliency = im.clone();
        image_saliency.convertTo(image_saliency, CV_32FC3);

        cv::Mat saliency_map = cv::Mat::zeros(image_saliency.size(), CV_32FC1);
        Saliency(image_saliency, saliency_map, 0.8f, 1.0f);

        std::vector<cv::Vec4f> matches_reserved;
        for (auto& iter : matches)
        {
            const int x = iter[0];
            const int y = iter[1];
            if (saliency_map.ptr<float>(y)[x] > params.saliency_th)
            {
                matches_reserved.emplace_back(iter);
            }
        }

        matches.clear();
        matches.assign(matches_reserved.begin(), matches_reserved.end());

        if (params.verbose) {
            std::cout << "Saliency filtering, remaining matches size: " << matches.size() << std::endl;
        }
    }

    // prepare variables
    const int nns = MY_MIN(params.nn, (int)matches.size());
    if (params.verbose) {
        std::cout << "Computing " << nns << " nearest neighbors for each match" << std::endl;
    }
    // compute nearest matches for each seed
    cv::Mat nnf = cv::Mat::zeros(matches.size(), nns, CV_32FC1);
    cv::Mat dis = cv::Mat::zeros(matches.size(), nns, CV_32FC1);
    cv::Mat labels = cv::Mat::zeros(edges.rows, edges.cols, CV_32FC1);

    dt_params_t dt_params{};
    dist_trf_nnfield_subset(nnf, dis, labels, matches, edges, dt_params, matches, n_thread);
}

