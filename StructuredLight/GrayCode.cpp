#include "GrayCode.h"

void GrayCode::GenBinaryCode()
{
	const int num = std::pow(2, bits);
	for (int index = 0; index < num; index++)
	{
		this->binary_code_.emplace(index, std::bitset<bits>(index));
	}
}

void GrayCode::GenGrayCode()
{
	const int num = std::pow(2, bits);
	for (int index = 0; index < num; index++)
	{
		std::bitset<bits> binary_code = std::bitset<bits>(index);
		std::bitset<bits> binary_code_right = std::bitset<bits>(index) >> 1;
		std::bitset<bits> gray_code = binary_code ^= binary_code_right;

		this->gray_code_.emplace(index, gray_code);
	}
}