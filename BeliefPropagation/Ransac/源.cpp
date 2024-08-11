#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>


struct float3
{
    float x, y, z;
};

struct plane
{
    float x, y, z, D;
};

inline void ImportPointCloud(char& filename, std::vector<float3>& points)
{
    float3 temp;
    bool mStatus = false;
    std::ifstream import;
    import.open(&filename);
    while (import.peek() != EOF)
    {
        import >> temp.x >> temp.y >> temp.z;
        points.push_back(temp);
    }
    import.close();
}

inline float DistanceToPlane(float3& point, plane& mPlane)
{
    return (mPlane.x * point.x + mPlane.y * point.y + mPlane.z * point.z + mPlane.D);
}

inline void PlaneFromPoints(std::vector<float3>& points, plane& mPlane)
{
    if (points.size() < 3) throw std::runtime_error("Not enough points to calculate plane");

    float3 sum = { 0,0,0 };

    for (int i = 0; i < static_cast<int>(points.size()); i++)
    {
        sum.x += points[i].x;
        sum.y += points[i].y;
        sum.z += points[i].z;
    }

    float3 centroid = { 0,0,0 };
    centroid.x = sum.x / float(points.size());
    centroid.y = sum.y / float(points.size());
    centroid.z = sum.z / float(points.size());


    float xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (int i = 0; i < static_cast<int>(points.size()); i++)
    {
        float3 temp;
        temp.x = points[i].x - centroid.x;
        temp.y = points[i].y - centroid.y;
        temp.z = points[i].z - centroid.z;

        xx += temp.x * temp.x;
        xy += temp.x * temp.y;
        xz += temp.x * temp.z;
        yy += temp.y * temp.y;
        yz += temp.y * temp.z;
        zz += temp.z * temp.z;
    }

    float detX = yy * zz - yz * yz;
    float detY = xx * zz - xz * xz;
    float detZ = xx * yy - xy * xy;

    float detMax = std::max(std::max(detX, detY), detZ);
    if (detMax <= 0)
    {
        mPlane.x = 0; mPlane.y = 0; mPlane.z = 0; mPlane.D = 0;
    }

    float3 dir{};
    if (detMax == detX)
    {
        float a = static_cast<float>((xz * yz - xy * zz) / detX);
        float b = static_cast<float>((xy * yz - xz * yy) / detX);
        dir = { 1.0, a, b };
    }
    else if (detMax == detY)
    {
        float a = static_cast<float>((yz * xz - xy * zz) / detY);
        float b = static_cast<float>((xy * xz - yz * xx) / detY);
        dir = { a, 1.0, b };
    }
    else
    {
        float a = static_cast<float>((yz * xy - xz * yy) / detZ);
        float b = static_cast<float>((xz * xy - yz * xx) / detZ);
        dir = { a, b, 1.0 };
    }

    float dirTemp = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    dir.x /= dirTemp;
    dir.y /= dirTemp;
    dir.z /= dirTemp;

    mPlane.x = dir.x;
    mPlane.y = dir.y;
    mPlane.z = dir.z;
    mPlane.D = -(mPlane.x * centroid.x + mPlane.y * centroid.y + mPlane.z * centroid.z);
}

void Ransac3DPlane(std::vector<float3>& inPoints, plane& outPlane)
{
    float minSamplePercent = 0.4f;  // Select the initial sample percentage
    int minSampleNum = static_cast<int>(float(inPoints.size()) * minSamplePercent);
    float maxSamplePercent = 0.9f;  // Maximum percentage control iteration stop
    int maxSampleNum = static_cast<int>(float(inPoints.size()) * maxSamplePercent);

    float distThreshold = 0.2f;
    int iterations = 0;
    int maxIterations = 30;
    std::vector<float3> maxPlanePoints; // Satisfy the maximum number of interior points of the plane model

    while (iterations < maxIterations || maxPlanePoints.size() < maxSampleNum)
    {
        iterations++;
        float3 ptTemp1, ptTemp2, ptTemp3;
        plane planeParame = { 0, 0, 0, 0 };
        std::vector<float3> points;
        std::vector<float3> tempPlanePoints;
        std::random_shuffle(inPoints.begin(), inPoints.end());
        for (int i = 0; i < minSampleNum; i++)
        {
            ptTemp1.x = inPoints[i].x;
            ptTemp1.y = inPoints[i].y;
            ptTemp1.z = inPoints[i].z;

            points.push_back(ptTemp1);
        }

        PlaneFromPoints(points, planeParame);
        points.clear();

        for (int i = 0; i < static_cast<int>(inPoints.size()); i++)
        {
            float distTemp;
            ptTemp2.x = inPoints[i].x;
            ptTemp2.y = inPoints[i].y;
            ptTemp2.z = inPoints[i].z;
            distTemp = DistanceToPlane(ptTemp2, planeParame);
            if (distTemp < distThreshold)
            {
                tempPlanePoints.push_back(ptTemp2);
            }
            else { continue; }
        }

        if (tempPlanePoints.size() > maxPlanePoints.size())
        {
            maxPlanePoints.clear();
            maxPlanePoints.reserve(tempPlanePoints.size());
            for (int i = 0; i < static_cast<int>(tempPlanePoints.size()); i++)
            {
                ptTemp3.x = tempPlanePoints[i].x;
                ptTemp3.y = tempPlanePoints[i].y;
                ptTemp3.z = tempPlanePoints[i].z;
                maxPlanePoints.push_back(ptTemp3);
            }
        }
        tempPlanePoints.clear();

    }

    PlaneFromPoints(maxPlanePoints, outPlane);
}

void test()
{
    char filename[255];
    sprintf_s(filename, 255, "111.asc");
    std::vector<float3> points;
    plane mPlane;
    ImportPointCloud(*filename, points);
    //PlaneFromPoints(points, mPlane);
    Ransac3DPlane(points, mPlane);
    std::cout << mPlane.x << "  " << mPlane.y << "  " << mPlane.z << "  " << mPlane.D << std::endl;

}

int main()
{
    test();
    return 0;
}
