#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "./func.hpp"

int main(int argc, char* argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // 初始化算法参数
    params_t params{};

    //std::filesystem::path file_path = "E:\\ExposureDetect\\imgs";
    //std::cout << "Traverse the imgs..." << std::endl;
    //for (const auto& iter : std::filesystem::directory_iterator(file_path))
    //{
    //    std::cout << "	-------------------------" << std::endl;
    //    std::cout << "	src path: " << iter.path().string() << std::endl;
    //    std::string img_path = iter.path().string();

    //    std::string overlap_path = ReplaceAll(iter.path().parent_path().string(), "imgs", "val4");
    //    overlap_path += "\\" + iter.path().stem().string() + "-heatmap" + iter.path().extension().string();
    //    std::cout << "	dst path: " << overlap_path << std::endl;

    //    // 读取图像
    //    cv::Mat src = cv::imread(img_path);
    //    if (src.empty()) { std::cerr << "Error, src is empty..."; }

    //    // 获取种子点
    //    std::vector<cv::Point2f> seeds{};
    //    getSeeds(seeds, src.cols, src.rows, params.step);

    //    // 获取score
    //    cv::Mat score{};
    //    getDescs(src, score, seeds, params.step, params.width);

    //    // 将heat map恢复到原始图像大小并进行后处理
    //    cv::Mat res_seg, res_fill, res_ana;
    //    recoverScore(seeds, score, src.size(), params.step);
    //    segmentMap(score, res_seg, params.thresh_low, params.thresh_high);
    //    fillHoles(res_seg, res_fill, params.erode_num, params.dilate_num);
    //    connectedDomainAnalysis(res_fill, res_ana);

    //    // 设置图像叠加显示
    //    cv::Mat heatmap{};
    //    score.convertTo(score, CV_8UC1, 1.0);
    //    cv::applyColorMap(score, heatmap, cv::COLORMAP_JET);

    //    float alpha = 0.5;
    //    cv::Mat overlap = src.clone();
    //    cv::rectangle(overlap, cv::Point2f(0, 0), cv::Point2f(src.cols, src.rows), (255, 0, 0), -1);
    //    cv::addWeighted(overlap, alpha, src, 1 - alpha, 0, overlap);
    //    cv::addWeighted(heatmap, alpha, src, 1 - alpha, 0, overlap);

    //    // cv::imwrite(overlap_path, score);
    //    // cv::imwrite(overlap_path, overlap);

    //    while (true) {
    //        cv::namedWindow("src", cv::WINDOW_NORMAL);
    //        cv::imshow("src", src);

    //        cv::namedWindow("overlap", cv::WINDOW_NORMAL);
    //        cv::imshow("overlap", overlap);

    //        cv::namedWindow("res_ana", cv::WINDOW_NORMAL);
    //        cv::imshow("res_ana", res_ana);

    //        //cv::imwrite("overlap.png", overlap);
    //        //cv::imwrite("res_ana.png", res_ana);

    //        if (cv::waitKey() == 13) { break; }
    //    }

    //    cv::destroyWindow("src");
    //    cv::destroyWindow("overlap");
    //    cv::destroyWindow("res_ana");
    //}

    // 测试单张图像

    std::string img_path = "E:\\ExposureDetect\\imgs\\test.jpg";
    std::string overlap_path = "E:\\ExposureDetect\\val4\\35.jpg";

    std::cout << "	-------------------------" << std::endl;
    std::cout << "	src path: " << img_path << std::endl;
    std::cout << "	dst path: " << overlap_path << std::endl;

    // 读取图像
    cv::Mat src = cv::imread(img_path);
    if (src.empty()) { std::cerr << "Error, src is empty..."; }

    // 获取种子点
    std::vector<cv::Point2f> seeds{};
    getSeeds(seeds, src.cols, src.rows, params.step);

    // 获取score
    cv::Mat score{};
    getDescs(src, score, seeds, params.step, params.width);

    // 将heat map恢复到原始图像大小并进行后处理
    cv::Mat res_seg, res_fill, res_ana;
    recoverScore(seeds, score, src.size(), params.step);
    segmentMap(score, res_seg, params.thresh_low, params.thresh_high);
    fillHoles(res_seg, res_fill, params.erode_num, params.dilate_num);
    connectedDomainAnalysis(res_fill, res_ana);

    // 设置图像叠加显示
    cv::Mat heatmap{};
    score.convertTo(score, CV_8UC1, 1.0);
    cv::applyColorMap(score, heatmap, cv::COLORMAP_JET);

    float alpha = 0.5;
    cv::Mat overlap = src.clone();
    cv::rectangle(overlap, cv::Point2f(0, 0), cv::Point2f(src.cols, src.rows), (255, 0, 0), -1);
    cv::addWeighted(overlap, alpha, src, 1 - alpha, 0, overlap);
    cv::addWeighted(heatmap, alpha, src, 1 - alpha, 0, overlap);

    cv::imwrite("score.png", res_ana);
    cv::imwrite("overlap.png", overlap);

    cv::namedWindow("src", cv::WINDOW_NORMAL);
    cv::imshow("src", src);

    cv::namedWindow("overlap", cv::WINDOW_NORMAL);
    cv::imshow("overlap", overlap);

    cv::namedWindow("res_ana", cv::WINDOW_NORMAL);
    cv::imshow("res_ana", res_ana);

    cv::waitKey();

    return 0;
}
