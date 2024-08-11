#ifndef EPIC_HPP
#define EPIC_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "./EpicAux.hpp"
#include "./Saliency.hpp"

/* parameter of epic */
struct epic_params_t
{
    std::string method = "LA";      // method for interpolation: la (locally-weighted affine) or nw (nadaraya-watson)
    float saliency_th = 0.045f;     // matches coming from pixels with a saliency below this threshold are removed before interpolation
    int pref_nn = 25;               // number of neighbors for consistent checking
    float pref_th = 5.0f;           // threshold for the first prefiltering step
    int nn = 160;                   // number of neighbors to consider for the interpolation
    float coef_kernel = 1.1f;       // coefficient in the sigmoid of the interpolation kernel
    float euc = 0.001f;             // constant added to the edge cost
    int verbose = 1;                // verbose mode
};


/// @brief main function for edge-preserving interpolation of correspondences
/// @param flowx: x-component of the flow (output)
/// @param flowy: y-component of the flow (output)
/// @param im: first image (in lab colorspace)
/// @param input_matches: input matches with each line a match and the first four columns containing x1 y1 x2 y2
/// @param edges: edges cost (can be modified)
/// @param params: parameters
/// @param n_thread: number of threads
/// @return None
void epic(cv::Mat& flowx, cv::Mat& flowy, const cv::Mat& im,
    const std::vector<cv::Vec4f>& input_matches, cv::Mat& edges,
    const epic_params_t& params, const int n_thread);

#endif // !EPIC_HPP
