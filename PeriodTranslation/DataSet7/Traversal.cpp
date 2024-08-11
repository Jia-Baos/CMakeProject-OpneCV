#include "Traverlsal.h"

/********************************************/
//�����븺������ļ����µ�����ͼƬ
/********************************************/

//���Ĵ���
void getFileNames(std::string path, std::vector<std::string>& files)
{
	//�ļ����
	//ע�⣺�ҷ�����Щ���´���˴���long���ͣ�ʵ�������лᱨ������쳣
	intptr_t hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,�ݹ����
			//�������,���ļ�����·������vector��
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
	//�Լ�ѡ��Ŀ¼����
	std::string path("D:/ProjectDALL/images");
	getFileNames(path, fileNames);
	//����·��path�µ�����ͼƬ
	for (const auto& img_path : fileNames)
	{
		std::cout << img_path << std::endl;
	}
}
