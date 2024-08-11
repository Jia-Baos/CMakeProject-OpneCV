#include "Traverlsal.h"

/********************************************/
//本代码负责遍历文件夹下的所有图片
/********************************************/

//核心代码
void getFileNames(std::string path, std::vector<std::string>& files)
{
	//文件句柄
	//注意：我发现有些文章代码此处是long类型，实测运行中会报错访问异常
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,递归查找
			//如果不是,把文件绝对路径存入vector中
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFileNames(p.assign(path).append("/").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


void getFileNames_test()
{
	std::vector<std::string> fileNames;
	//自己选择目录测试
	std::string path("D:/ProjectDALL/images");
	getFileNames(path, fileNames);
	//遍历路径path下的所有图片
	for (const auto& img_path : fileNames)
	{
		std::cout << img_path << std::endl;
	}
}
