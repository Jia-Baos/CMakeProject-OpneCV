#include "BeliefPropagation.h"

BeliefPropagation::BeliefPropagation()
{
	// the width_search_ equals half of width_blocks is advocated
	this->width_blocks_ = 32;
	this->width_search_ = 16;
	std::cout << "Initalize the Class: BeliefPropagation" << std::endl;
}


BeliefPropagation::~BeliefPropagation()
{
	std::cout << "Destory the Class: BeliefPropagation" << std::endl;
}


bool BeliefPropagation::ImagePadded(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output1, cv::Mat& output2)
{
	int rows_max = input1.rows > input2.rows ? input1.rows : input2.rows;
	int	cols_max = input1.cols > input2.cols ? input1.cols : input2.cols;

	// Pad the image avoid the block's size euqals 1
	int rows_max_even = width_blocks_ * (static_cast<int>(rows_max / width_blocks_) + 1);
	int cols_max_even = width_blocks_ * (static_cast<int>(cols_max / width_blocks_) + 1);

	cv::copyMakeBorder(input1, output1, 0, rows_max_even - input1.rows, 0,
		cols_max_even - input1.cols, cv::BORDER_REPLICATE);
	cv::copyMakeBorder(input2, output2, 0, rows_max_even - input2.rows, 0,
		cols_max_even - input2.cols, cv::BORDER_REPLICATE);

	return 0;
}


bool  BeliefPropagation::SetInput(const cv::Mat& fixed_image, const cv::Mat& moved_image)
{
	// Assert the src has read success and has the same size
	assert(!fixed_image.empty());
	assert(!moved_image.empty());

	// Revise the size of input image if the size is different
	cv::Mat fixed_image_padded, moved_image_padded;
	if (fixed_image.size() != moved_image.size())
	{
		ImagePadded(fixed_image, moved_image, fixed_image_padded, moved_image_padded);
	}
	else
	{
		fixed_image_padded = fixed_image;
		moved_image_padded = moved_image;
	}

	// Assert the type of src is CV_8UC1
	if (fixed_image_padded.type() == CV_8UC3 || moved_image_padded.type() == CV_8UC3)
	{
		cv::cvtColor(fixed_image_padded, this->fixed_image_, cv::COLOR_BGR2GRAY);
		cv::cvtColor(moved_image_padded, this->moved_image_, cv::COLOR_BGR2GRAY);
	}
	else {
		this->fixed_image_ = fixed_image;
		this->moved_image_ = moved_image;
	}

	this->col_nums_ = ((fixed_image_.cols % width_blocks_) == 0) ?
		(fixed_image_.cols / width_blocks_) : (fixed_image_.cols / width_blocks_) + 1;
	this->row_nums_ = ((fixed_image_.rows % width_blocks_) == 0) ?
		(fixed_image_.rows / width_blocks_) : (fixed_image_.rows / width_blocks_) + 1;

	return 0;
}


bool BeliefPropagation::Compute(const cv::Mat& fixed_image, const cv::Mat& moved_image)
{
	// Construct the fixed_image_ and moved_image_
	SetInput(fixed_image, moved_image);

	// Initalize the blocks_offset_
	cv::Mat fixed_image_current = this->fixed_image_;
	cv::Mat moved_image_current = this->moved_image_;
	cv::Mat blocks_offset_ = cv::Mat::zeros(this->row_nums_, this->col_nums_, CV_32FC2);

	// Initalize the MarkovRandomField
	MarkovRandomField mrf_x, mrf_y;
	InitialMarkovRandomField(fixed_image_current,
		moved_image_current, blocks_offset_, mrf_x, mrf_y);

	// Send message
	for (int i = 0; i < Iteration; i++)
	{
		BP(mrf_x, RIGHT);
		BP(mrf_x, LEFT);
		BP(mrf_x, UP);
		BP(mrf_x, DOWN);

		BP(mrf_y, RIGHT);
		BP(mrf_y, LEFT);
		BP(mrf_y, UP);
		BP(mrf_y, DOWN);

		unsigned int energy_x = MAP(mrf_x);
		unsigned int energy_y = MAP(mrf_y);

		std::cout << "Iteration: " << i << "  Energy_x: " << energy_x << std::endl;
		std::cout << "Iteration: " << i << "  Energy_y: " << energy_y << std::endl;
	}

	cv::Mat output_x = cv::Mat::zeros(mrf_x.height, mrf_x.width, CV_8UC1);
	cv::Mat output_y = cv::Mat::zeros(mrf_y.height, mrf_y.width, CV_8UC1);
	for (int i = LABELS; i < mrf_x.height - LABELS; i++)
	{
		for (int j = LABELS; j < mrf_x.width - LABELS; j++)
		{
			output_x.at<uchar>(i, j) = mrf_x.grid[i * mrf_x.width + j].best_assignment * (256 / LABELS);
			output_y.at<uchar>(i, j) = mrf_y.grid[i * mrf_y.width + j].best_assignment * (256 / LABELS);
		}
	}

	cv::namedWindow("Output_x", cv::WINDOW_NORMAL);
	imshow("Output_x", output_x);
	cv::namedWindow("Output_y", cv::WINDOW_NORMAL);
	imshow("Output_y", output_y);
	cv::imwrite("output_x.png", output_x);
	cv::imwrite("output_y.png", output_y);
	return 0;
}


void BeliefPropagation::InitialMarkovRandomField(const cv::Mat& fixed_image_current,
	const cv::Mat& moved_image_current,
	const cv::Mat& blocks_offset,
	MarkovRandomField& mrf_x,
	MarkovRandomField& mrf_y)
{
	mrf_x.height = row_nums_;
	mrf_x.width = col_nums_;
	mrf_y.height = row_nums_;
	mrf_y.width = col_nums_;
	int node_num = row_nums_ * col_nums_;
	mrf_x.grid.resize(node_num);
	mrf_y.grid.resize(node_num);

	for (int i = 0; i < node_num; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			for (int k = 0; k < LABELS; k++)
			{
				mrf_x.grid[i].msg[j][k] = 0;
				mrf_y.grid[i].msg[j][k] = 0;
			}
		}
	}

	for (int i = 0; i < row_nums_; i++)
	{
		for (int j = 0; j < col_nums_; j++)
		{
			int offset_x_current = OFFSET_MIN;
			int offset_y_current = OFFSET_MIN;
			for (int k = 0; k < LABELS; k++)
			{
				mrf_x.grid[i * col_nums_ + j].msg[DATA][k] =
					DataCost(fixed_image_current, moved_image_current,
						i, j, offset_x_current, HORINZONTAL);
				mrf_y.grid[i * col_nums_ + j].msg[DATA][k] =
					DataCost(fixed_image_current, moved_image_current,
						i, j, offset_y_current, VERTICAL);
				offset_x_current = offset_x_current + STEP;
				offset_y_current = offset_y_current + STEP;
			}
		}
	}
}


unsigned int BeliefPropagation::DataCost(const cv::Mat& fixed_image_current,
	const cv::Mat& moved_image_current,
	int i, int j, int offset, DIRECTION direction)
{
	cv::Mat fixed_image_temp = fixed_image_current.clone();
	cv::Mat moved_image_temp = moved_image_current.clone();
	int padded_distance = abs(offset);
	cv::copyMakeBorder(moved_image_temp, moved_image_temp, padded_distance, padded_distance,
		padded_distance, padded_distance, cv::BORDER_REPLICATE);

	cv::Mat region1, region2;
	if (direction == HORINZONTAL)
	{
		region1 = fixed_image_temp(cv::Rect(j * width_blocks_,
			i * width_blocks_, width_blocks_, width_blocks_));
		region2 = moved_image_temp(cv::Rect(padded_distance + j * width_blocks_ + offset,
			padded_distance + i * width_blocks_, width_blocks_, width_blocks_));
	}

	if (direction == VERTICAL)
	{
		region1 = fixed_image_temp(cv::Rect(j * width_blocks_,
			i * width_blocks_, width_blocks_, width_blocks_));
		region2 = moved_image_temp(cv::Rect(padded_distance + j * width_blocks_,
			padded_distance + i * width_blocks_ + offset, width_blocks_, width_blocks_));
	}

	cv::Mat mse_matrix;
	cv::absdiff(region1, region2, mse_matrix);

	float mse_matrix_sum = cv::sum(mse_matrix)[0];
	unsigned int mse_matrix_average = mse_matrix_sum / (mse_matrix.cols * mse_matrix.rows);

	return mse_matrix_average;
}


void BeliefPropagation::BP(MarkovRandomField& mrf,ATTRIBUTE attribute)
{
	int width = mrf.width;
	int height = mrf.height;

	switch (attribute)
	{
	case RIGHT:
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width - 1; j++)
			{
				SendMsg(mrf, i, j, attribute);
			}
		}
		break;

	case LEFT:
		for (int i = 0; i < height; i++)
		{
			for (int j = width - 1; j > 0; j--)
			{
				SendMsg(mrf, i, j, attribute);
			}
		}
		break;

	case UP:
		for (int i = 0; i < width; i++)
		{
			for (int j = height - 1; j > 0; j--)
			{
				SendMsg(mrf, j, i, attribute);
			}
		}
		break;

	case DOWN:
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height - 1; j++)
			{
				SendMsg(mrf, j, i, attribute);
			}
		}
		break;

	default:
		break;
	}
}
 

#define min(a,b) (a<b)?a:b

unsigned int BeliefPropagation::SmoothCost(int i, int j)
{
	int d = (2 * i + OFFSET_MIN) - (2 * j + OFFSET_MIN);
	return LAMBDA * min(abs(d), SMOOTHNESS_TRUC);
}


void BeliefPropagation::SendMsg(MarkovRandomField& mrf, int x, int y, ATTRIBUTE attribute)
{
	unsigned int new_msg[LABELS];
	int width = mrf.width;

	//collect msg and pass to the next node
	for (int i = 0; i < LABELS; i++)
	{
		unsigned int min_val = UINT_MAX;
		for (int j = 0; j < LABELS; j++)
		{
			unsigned int p = 0;
			p += SmoothCost(i, j);
			p += mrf.grid[x * width + y].msg[DATA][j];

			if (attribute != LEFT)
			{
				p += mrf.grid[x * width + y].msg[LEFT][j];
			}
			if (attribute != RIGHT)
			{
				p += mrf.grid[x * width + y].msg[RIGHT][j];
			}
			if (attribute != UP)
			{
				p += mrf.grid[x * width + y].msg[UP][j];
			}
			if (attribute != DOWN)
			{
				p += mrf.grid[x * width + y].msg[DOWN][j];
			}
			min_val = min(min_val, p);
		}
		new_msg[i] = min_val;
	}

	//update each node's msg
	for (int i = 0; i < LABELS; i++)
	{
		switch (attribute)
		{
		case LEFT:
			mrf.grid[x * width + y - 1].msg[RIGHT][i] = new_msg[i];
			break;

		case RIGHT:
			mrf.grid[x * width + y + 1].msg[LEFT][i] = new_msg[i];
			break;

		case UP:
			mrf.grid[(x - 1) * width + y].msg[DOWN][i] = new_msg[i];
			break;

		case DOWN:
			mrf.grid[(x + 1) * width + y].msg[UP][i] = new_msg[i];
			break;

		default:
			break;
		}
	}
}
 

unsigned int BeliefPropagation::MAP(MarkovRandomField& mrf)
{
	for (int i = 0; i < mrf.grid.size(); i++)
	{
		unsigned int best = UINT_MAX;
		for (int j = 0; j < LABELS; j++)
		{
			unsigned int cost = 0;
			cost += mrf.grid[i].msg[LEFT][j];
			cost += mrf.grid[i].msg[RIGHT][j];
			cost += mrf.grid[i].msg[UP][j];
			cost += mrf.grid[i].msg[DOWN][j];

			if (cost < best) {
				best = cost;
				mrf.grid[i].best_assignment = j;
			}
		}
	}

	int width = mrf.width;
	int height = mrf.height;

	unsigned int energy = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int cur_label = mrf.grid[i * width + j].best_assignment;
			energy += mrf.grid[i * width + j].msg[DATA][cur_label];
			if (j - 1 >= 0)
			{
				energy += SmoothCost(cur_label, mrf.grid[i * width + j - 1].best_assignment);
			}
			if (j + 1 < width)
			{
				energy += SmoothCost(cur_label, mrf.grid[i * width + j + 1].best_assignment);
			}
			if (i - 1 >= 0)
			{
				energy += SmoothCost(cur_label, mrf.grid[(i - 1) * width + j].best_assignment);
			}
			if (i + 1 < height)
			{
				energy += SmoothCost(cur_label, mrf.grid[(i + 1) * width + j].best_assignment);
			}
		}
	}

	return energy;
}