#include "BeliefPropagation.h"

BeliefPropagation::BeliefPropagation()
{
	std::cout << "Initalize the Class: BeliefPropagation" << std::endl;
}

BeliefPropagation::~BeliefPropagation()
{
	std::cout << "Destory the Class: BeliefPropagation" << std::endl;
}


BeliefPropagation::BeliefPropagation(cv::Mat lImage, cv::Mat rImage)
{
	leftImage = lImage;
	rightImage = rImage;
	leftGray = lImage;
	rightGray = rImage;
	std::cout << "Initalize the Class: BeliefPropagation" << std::endl;
}

void BeliefPropagation::initNode(Node_s* node, int i, int j)
{
	node->x = i;
	node->y = j;
	node->up = NULL;
	node->down = NULL;
	node->left = NULL;
	node->right = NULL;
	memset(node->messageUp, 0, sizeof(int) * L);
	memset(node->messageDown, 0, sizeof(int) * L);
	memset(node->messageLeft, 0, sizeof(int) * L);
	memset(node->messageRight, 0, sizeof(int) * L);
}

//用链表表示图
void BeliefPropagation::generateList()
{
	head = (struct Node_s*)malloc(sizeof(Node_s));
	Node_s* pre_col = head;
	int height = leftGray.rows;
	int width = leftGray.cols;

	initNode(head, 0, 0);

	//建立第一列的链表
	for (int i = 1; i < height; i++)
	{
		Node_s* node = (struct Node_s*)malloc(sizeof(Node_s));
		initNode(node, i, 0);
		pre_col->down = node;
		node->up = pre_col;
		pre_col = pre_col->down;
	}

	//建立第一行的链表
	Node_s* pre_row = head;
	for (int j = 1; j < width; j++)
	{
		Node_s* node = (struct Node_s*)malloc(sizeof(Node_s));
		initNode(node, 0, j);
		pre_row->right = node;
		node->left = pre_row;
		pre_row = pre_row->right;
	}

	Node_s* preR = head->right;
	Node_s* preC = head->down;
	Node_s* pr = preR;
	Node_s* pc = preC;
	for (int i = 1; i < height; i++)
	{
		pc = preC;
		pr = preR;
		for (int j = 1; j < width; j++)
		{
			Node_s* node = (struct Node_s*)malloc(sizeof(Node_s));
			initNode(node, i, j);
			pc->right = node;
			node->left = pc;
			pr->down = node;
			node->up = pr;
			pr = pr->right;
			pc = pc->right;
		}
		preC = preC->down;
		preR = preR->down;
	}
}

void BeliefPropagation::printList()
{
	Node_s* p = head;
	Node_s* d = p;
	Node_s* r = p;

	while (d)
	{
		while (r)
		{
			std::cout << "x: " << r->x << "; y: " << r->y << std::endl;
			r = r->right;
		}
		d = d->down;
		r = d;
	}

}


int BeliefPropagation::Cost(int li, int lj, int x)
{
	int  f = abs(leftGray.at<uchar>(li, lj) - rightGray.at<uchar>(li, lj - x));
	return f;
}


void BeliefPropagation::computeME(Node_s* node, Node_s* e, int t)
{
	for (int j = 0; j < L; j++)
	{
		int minsum = INTMAX_MAX;
		for (int i = 0; i < L; i++)
		{
			if ((node->y - i) < 0) break;

			int m = 0;
			int Di = Cost(node->x, node->y, i);
			int Vij = C * abs(i - j);

			Node_s* li = node->left;
			if (e != li && li != NULL) m += li->messageRight[t - 1][i];

			Node_s* ri = node->right;
			if (e != ri && ri != NULL) m += ri->messageLeft[t - 1][i];

			Node_s* dow = node->down;
			if (e != dow && dow != NULL) m += dow->messageUp[t - 1][i];

			Node_s* u = node->up;
			if (e != u && u != NULL) m += u->messageDown[t - 1][i];

			int sum = m + Di + Vij;
			if (minsum > sum) minsum = sum;
		}

		if (e == node->right)  node->messageRight[t][j] = minsum;
		if (e == node->left)  node->messageLeft[t][j] = minsum;
		if (e == node->up)  node->messageUp[t][j] = minsum;
		if (e == node->down)  node->messageDown[t][j] = minsum;

	}
}

// 计算消息矩阵，T为迭代次数
void BeliefPropagation::computeM()
{
	int height = leftGray.rows;
	int width = leftGray.cols;

	for (int t = 0; t < T; t++)
	{
		std::cout << "Iter: " << t << std::endl;
		Node_s* p = head;
		Node_s* d = p;
		Node_s* r = p;

		while (d)
		{
			while (r)
			{
				/*if (!(r->up && r->down && r->left && r->right)) {
					r = r->right;
					continue;
				}*/

				//对于每一个节点分别使他对周围的邻居节点更新消息
				if (r->up) computeME(r, r->up, t);
				if (r->down) computeME(r, r->down, t);
				if (r->left) computeME(r, r->left, t);
				if (r->right) computeME(r, r->right, t);
				r = r->right;
			}
			d = d->down;
			r = d;
		}
	}
}

//找到使得节点n置信度最小的视差x
int BeliefPropagation::computeDisparityByBelief(Node_s* n)
{
	int minx = L;
	int minBelief = INTMAX_MAX;
	for (int i = 0; i < L; i++)
	{
		if ((n->y - i) < 0) break;
		int Di = Cost(i, n->x, n->y);
		int message = 0;
		if (n->up)  message += n->up->messageDown[T - 1][i];
		if (n->down) message += n->down->messageUp[T - 1][i];
		if (n->left) message += n->left->messageRight[T - 1][i];
		if (n->right) message += n->right->messageLeft[T - 1][i];
		int sum = Di + message;
		if (minBelief > sum) minBelief = sum, minx = i;
	}
	return minx;
}

//生成视差图
void  BeliefPropagation::generateDisparity()
{
	Node_s* p = head;
	Node_s* r = p;
	Node_s* d = p;
	disparity = cv::Mat::zeros(leftGray.size(), CV_8UC1);
	while (d)
	{
		while (r)
		{
			/*if (!(r->up && r->down && r->left &&r->right))
			{
				r= r->right;
				continue;
			}*/

			//对于每一个节点根据周围的消息，找到使置信度最小的视差
			int d = computeDisparityByBelief(r);
			disparity.at<uchar>(r->x, r->y) = d * (256 / L);
			std::cout << "x: " << r->x << "; y: " << r->y << std::endl;

			r = r->right;
		}
		d = d->down;
		r = d;
	}
}
