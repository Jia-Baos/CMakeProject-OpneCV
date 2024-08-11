#include <iostream>
#include <string>
#include <vector>

#include "BP.h"

int main(int argc, char* argv[])
{
	const std::string s1 = "D:/Code-VS/picture/test-data/teddy/im6.png";
	const std::string s2 = "D:/Code-VS/picture/test-data/teddy/im2.png";
	//cv::Mat lImg = cv::imread(argv[1], 0); //left image
	//cv::Mat rImg = cv::imread(argv[2], 0); //right image
	cv::Mat lImg = cv::imread(s1, 0); //left image
	cv::Mat rImg = cv::imread(s2, 0); //right image


	//cv::imshow("test",lImg);
	//cv::waitKey();

	//int baseDist = atoi(argv[3]); //baseline distance
	
	int baseDist = 8;
	BP myBP(lImg, rImg, baseDist);

	myBP.loopyBPIterate();

	return 0;
}
