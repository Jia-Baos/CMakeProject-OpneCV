#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "./Epic.hpp"
#include "./GeodesicDistance.hpp"
#include "./Saliency.hpp"

int main(int argc, char* argv[])
{
    std::cout << "Version: " << CV_VERSION << std::endl;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    epic_params_t epic_params{};

    std::string image_path = "D:/Code-VS/CMakeProject-EpicFlow/frame_0029.png";
    std::string match_path = "D:/Code-VS/CMakeProject-EpicFlow/frame_0029.txt";
    std::string model_path = "D:/Code-VS/CMakeProject-EpicFlow/model.yml";

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2Lab);

    // Read matches
    std::cout << "Read matches..." << std::endl;
    std::vector<cv::Vec4f> input_matches;

    FILE* file;
    float x1 = 0.0;
    float y1 = 0.0;
    float x2 = 0.0;
    float y2 = 0.0;
    fopen_s(&file, match_path.c_str(), "r");
    while (!feof(file) &&
        fscanf_s(file, "%f %f %f %f%*[^\n]", &x1, &y1, &x2, &y2) == 4) {
        input_matches.emplace_back(cv::Vec4f(x1, y1, x2, y2));
    }

    // Compute cost map by SED
    std::cout << "Read edges..." << std::endl;
    cv::Mat image_sed = image.clone();
    image_sed.convertTo(image_sed, CV_32FC3, 1 / 255.0);
    cv::Mat edges = cv::Mat::zeros(image_sed.size(), CV_32FC1);
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar =
        cv::ximgproc::createStructuredEdgeDetection(model_path);
    pDollar->detectEdges(image_sed, edges);

    cv::Mat flowx = cv::Mat::zeros(image.size(), CV_32FC1);
    cv::Mat flowy = cv::Mat::zeros(image.size(), CV_32FC1);
    epic(flowx, flowy, image, input_matches, edges, epic_params, 4);

    return 0;
}
