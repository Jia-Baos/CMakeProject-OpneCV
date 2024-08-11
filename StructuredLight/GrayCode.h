#pragma once

#include <map>
#include <bitset>
#include <opencv2/opencv.hpp>

const int bits = 5;

class GrayCode
{
public:
	std::map<int, std::bitset<bits>> binary_code_;
	std::map<int, std::bitset<bits>> gray_code_;

	GrayCode() = default;
	~GrayCode() = default;
	GrayCode(const GrayCode& gray_code) = delete;
	GrayCode& operator=(const GrayCode& gray_code) = delete;

	void GenBinaryCode();
	void GenGrayCode();
};
