#pragma once
#ifndef TRAVERSAL_H
#define TRAVERSAL_H

#pragma once
# include<iostream>
# include<string>
# include<vector>
//注意这个头文件
# include<io.h>

/********************************************/
//本代码负责遍历文件夹下的所有图片
/********************************************/

//核心代码
void getFileNames(std::string path, std::vector<std::string>& files);

void getFileNames_test();

#endif // !TRAVERSAL_H