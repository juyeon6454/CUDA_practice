#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <io.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ctime>

using namespace std;
using namespace std::filesystem;

typedef unsigned short ushort;

void dirDarkfile();
void dirGainfile();
void darkMap();
void gainMap();

vector<string> darkfn; 
vector<string> gainfn; 
vector<string> originfn;

void dirOriginfile();
void callibration();


__global__ void cudaFilesSum(ushort* inimg, float* averageimg, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + (y * width);

    if (x < width && y < height) {
        *(averageimg + offset) += *(inimg + offset);
    }
}

__global__ void cudaPixelAvg(float* averageimg, ushort* outimg, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + (y * width);

    if (x < width && y < height) {
        *(outimg + offset) = *(averageimg + offset) / 101;
    }
}

__global__ void cudaCalibration(ushort* darkMapImg, ushort* GainMapImg,
    ushort* MTF_VImg, double subGainAvg,
    int width, int height, ushort* outimg)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + (y * width);

    if (x < width && y < height) {
        *(outimg + offset) =
            (ushort)(abs(*(MTF_VImg + offset) - (*(darkMapImg + offset))) / (float)((*(GainMapImg + offset)) - (*(darkMapImg + offset))) * subGainAvg);
    }
}

int main(int argc, char* argv[])
{
    clock_t start, end;
    double result;
    start = clock();

    dirDarkfile();
    dirGainfile();
    darkMap();
    gainMap();

    dirOriginfile();
    callibration();

    end = clock();
    result = (double)(end - start);
    cout << " 수행 시간 : " << result << "millisecond" << "\n";

    printf("Success processing !!\n");

    return 0;
}

void darkMap() {

    FILE* infp, * outfp;
    //string file = "C:/calibration_mentoring/pano/DarkMap(pano).raw";
    ushort* inimg, * outimg;

    float* f_averageimg;
    int width = 80, height = 1628;
    int imageSize = width * height;

    inimg = (ushort*)malloc(sizeof(ushort) * imageSize);
    outimg = (ushort*)malloc(sizeof(ushort) * imageSize);
    f_averageimg = (float*)malloc(sizeof(float) * imageSize);

    memset(inimg, 0, sizeof(ushort) * imageSize);
    memset(outimg, 0, sizeof(ushort) * imageSize);
    memset(f_averageimg, 0, sizeof(float) * imageSize);

    float* d_averageimg;
    cudaMalloc(&d_averageimg, sizeof(float) * imageSize);
    cudaMemset(d_averageimg, 0, sizeof(float) * imageSize);

    const dim3 dimGrid((int)ceil((width / 4)), (int)ceil((height) / 4));
    const dim3 dimBlock(4, 4);

    ushort* d_outimg;
    cudaMalloc(&d_outimg, sizeof(ushort) * imageSize);

    /* cuda reset */
    ushort* d_inimg = NULL;
    cudaMalloc(&d_inimg, sizeof(ushort) * imageSize);

    /* 파일 cnt로 돌면서 구간별로 합산해서 평균 내도록 수정 */
    vector<string>::iterator iter;
    iter = darkfn.begin();
    for (iter = darkfn.begin(); iter != darkfn.end(); iter++) {

        memset(inimg, 0, sizeof(ushort) * imageSize);
        string file2 = *iter;
        int cnt = 0;

        //cout << "path: " << file2 << endl;   //      fopen Ȯ

        if ((infp = fopen(file2.c_str(), "rb")) == NULL) {
            printf("%d No such file or folder\n", __LINE__);
            return;
        }

        fread(inimg, sizeof(ushort) * imageSize, 1, infp); 

        cudaMemset(d_inimg, 0, sizeof(ushort) * imageSize);
        cudaMemcpy(d_inimg, inimg, sizeof(ushort) * imageSize, cudaMemcpyHostToDevice);

        cudaFilesSum << <dimGrid, dimBlock >> > (d_inimg, d_averageimg, width, height);

        cudaFree(d_inimg);

        fclose(infp);

        cudaMemset(d_outimg, 0, sizeof(ushort) * imageSize);

        if (cnt = 192 )
        {
            cudaPixelAvg << <dimGrid, dimBlock >> > (d_averageimg, d_outimg, width, height);

            cudaMemcpy(outimg, d_outimg, sizeof(ushort) * imageSize, cudaMemcpyDeviceToHost);

            cudaFree(d_outimg);
            cudaFree(d_averageimg);

            if ((outfp = fopen("C:/calibration_mentoring/pano/DarkMap(pano)1.raw", "wb")) == NULL) {
                printf("%d No such file or folder\n", __LINE__);
                return;
            }

            fwrite(outimg, sizeof(ushort) * imageSize, 1, outfp);
        }

        else if (cnt = )

        cudaMemset(d_averageimg, 0, sizeof(ushort) * imageSize);
        cnt++;
    }//iter 의 끝

    free(inimg);
    free(outimg);
    free(f_averageimg);
    fclose(outfp);
}

void gainMap() {

    FILE* infp, * outfp;
    string file = "C:/calibration_mentoring/pano/GainMap(pano).raw";

    ushort* inimg, * outimg;
    float* f_averageimg;
    int width = 80, height = 1628;
    int imageSize = width * height;

    inimg = (ushort*)malloc(sizeof(ushort) * imageSize);
    outimg = (ushort*)malloc(sizeof(ushort) * imageSize);
    f_averageimg = (float*)malloc(sizeof(float) * imageSize);

    memset(inimg, 0, sizeof(ushort) * imageSize);
    memset(outimg, 0, sizeof(ushort) * imageSize);
    memset(f_averageimg, 0, sizeof(float) * imageSize);

    float* d_averageimg;
    cudaMalloc(&d_averageimg, sizeof(float) * imageSize);
    cudaMemset(d_averageimg, 0, sizeof(float) * imageSize);

    const dim3 dimGrid((int)ceil((width / 4)), (int)ceil((height) / 4));
    const dim3 dimBlock(4, 4);

    vector<string>::iterator iter;
    iter = gainfn.begin();
    for (iter = gainfn.begin(); iter != gainfn.end(); iter++) {

        memset(inimg, 0, sizeof(ushort) * imageSize);
        string file2 = *iter;

        // cout << "path: " << file2 << endl;   //      fopen Ȯ  

        if ((infp = fopen(file2.c_str(), "rb")) == NULL) {
            printf("%d No such file or folder\n", __LINE__);
            return;
        }

        fread(inimg, sizeof(ushort) * imageSize, 1, infp);
        fclose(infp);

        ushort* d_inimg = NULL;
        cudaMalloc(&d_inimg, sizeof(ushort) * imageSize);
        cudaMemset(d_inimg, 0, sizeof(ushort) * imageSize);
        cudaMemcpy(d_inimg, inimg, sizeof(ushort) * imageSize, cudaMemcpyHostToDevice);


        cudaFilesSum << <dimGrid, dimBlock >> > (d_inimg, d_averageimg, width, height);
        cudaFree(d_inimg);

    }

    ushort* d_outimg;
    cudaMalloc(&d_outimg, sizeof(ushort) * imageSize);
    cudaMemset(d_outimg, 0, sizeof(ushort) * imageSize);

    cudaPixelAvg << <dimGrid, dimBlock >> > (d_averageimg, d_outimg, width, height);

    cudaMemcpy(outimg, d_outimg, sizeof(ushort) * imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_outimg);
    cudaFree(d_averageimg);

    if ((outfp = fopen(file.c_str(), "wb")) == NULL) {
        printf("%d No such file or folder\n", __LINE__);
        return;
    }

    fwrite(outimg, sizeof(ushort) * imageSize, 1, outfp);

    free(inimg);
    free(outimg);
    free(f_averageimg);
    fclose(outfp);
}

void callibration() {

    FILE* GainMapFp, * darkMapFp, * MTF_VFp, * outfp;
    int width = 80, height = 1628;
    int imageSize = width * height;
    int subImageSize = width * height;
    double subGainSum = 0, subGainAvg = 0;
    ushort* darkMapImg, * GainMapImg, * MTF_VImg, * outimg;

    darkMapImg = (ushort*)malloc(sizeof(ushort) * imageSize);
    GainMapImg = (ushort*)malloc(sizeof(ushort) * imageSize);
    MTF_VImg = (ushort*)malloc(sizeof(ushort) * imageSize);
    outimg = (ushort*)malloc(sizeof(ushort) * imageSize);

    memset(darkMapImg, 0, sizeof(ushort) * imageSize);
    memset(GainMapImg, 0, sizeof(ushort) * imageSize);


    if ((darkMapFp = fopen("C:/calibration_mentoring/pano/DarkMap(pano).raw", "rb")) == NULL) {
        printf("%d No such file or folder\n", __LINE__);
        return;
    }
    if ((GainMapFp = fopen("C:/calibration_mentoring/pano/GainMap(pano).raw", "rb")) == NULL) {
        printf("%d No such file or folder\n", __LINE__);
        return;
    }

    fread(darkMapImg, sizeof(ushort) * imageSize, 1, darkMapFp);
    fread(GainMapImg, sizeof(ushort) * imageSize, 1, GainMapFp);

    const dim3 dimGrid((int)ceil((width / 4)), (int)ceil((height) / 4));
    const dim3 dimBlock(4, 4);

    /* cuda reset */
    ushort* d_MTF_VImg, * d_outimg, * d_darkMapImg, * d_GainMapImg;

    cudaMalloc(&d_darkMapImg, sizeof(ushort) * imageSize);
    cudaMalloc(&d_GainMapImg, sizeof(ushort) * imageSize);

    cudaMemset(d_darkMapImg, 0, sizeof(ushort) * imageSize);
    cudaMemset(d_GainMapImg, 0, sizeof(ushort) * imageSize);

    cudaMemcpy(d_darkMapImg, darkMapImg, sizeof(ushort) * imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_GainMapImg, GainMapImg, sizeof(ushort) * imageSize, cudaMemcpyHostToDevice);


    vector<string>::iterator iter;
    iter = originfn.begin();
    for (iter = originfn.begin(); iter != originfn.end(); iter++) {

        memset(MTF_VImg, 0, sizeof(ushort) * imageSize);
        memset(outimg, 0, sizeof(ushort) * imageSize);

        string origin_file = *iter;

        if ((MTF_VFp = fopen(origin_file.c_str(), "rb")) == NULL) {
            printf("%d No such file or folder\n", __LINE__);
            return;
        }

        cout << "origin file : " << *iter << endl;

        fread(MTF_VImg, sizeof(ushort) * imageSize, 1, MTF_VFp);

        fclose(MTF_VFp);

        cudaMalloc(&d_MTF_VImg, sizeof(ushort) * imageSize);
        cudaMemset(d_MTF_VImg, 0, sizeof(ushort) * imageSize);
        cudaMemcpy(d_MTF_VImg, MTF_VImg, sizeof(ushort) * imageSize, cudaMemcpyHostToDevice);

        cudaMalloc(&d_outimg, sizeof(ushort) * imageSize);
        cudaMemset(d_outimg, 0, sizeof(ushort) * imageSize);


        subGainAvg = 0;
        subGainSum = 0;

        for (int i = 0; i < imageSize; i++) {

            subGainSum += GainMapImg[i];
        }

        subGainAvg = subGainSum / subImageSize;

        cudaCalibration << <dimGrid, dimBlock >> > (d_darkMapImg, d_GainMapImg, d_MTF_VImg, subGainAvg, width, height, d_outimg);

        cudaMemcpy(outimg, d_outimg, sizeof(ushort) * imageSize, cudaMemcpyDeviceToHost);

        string pullPath = *iter;
        int find = pullPath.rfind("\\") + 1;
        string fileName = pullPath.substr(find, pullPath.length() - find);

        //cout << "File Name : " << fileName << endl;

        cudaFree(d_MTF_VImg);
        cudaFree(d_outimg);

        string file = "C:/calibration_mentoring/pano/Result/ " + fileName;

        if ((outfp = fopen(file.c_str(), "wb")) == NULL) {
            printf("%d No such file or folder\n", __LINE__);
            return;
        }

        fwrite(outimg, sizeof(ushort) * imageSize, 1, outfp);

        fclose(outfp);
    }

    cudaFree(d_darkMapImg);
    cudaFree(d_GainMapImg);

    fclose(darkMapFp);
    fclose(GainMapFp);

    free(GainMapImg);
    free(darkMapImg);
    free(MTF_VImg);
    free(outimg);
}




void dirDarkfile()
{
    string file = "C:/calibration_mentoring/pano/Dark";

    for (const auto& file : directory_iterator(file)) {
        std::string filepath{ file.path().u8string() };
        darkfn.push_back(filepath);
    }

    return;
}

void dirGainfile()
{
    string file = "C:/calibration_mentoring/pano/Gain_No_Filter/80kV_10mA";

    for (const auto& file : directory_iterator(file)) {
        std::string filepath{ file.path().u8string() };
        gainfn.push_back(filepath);

        //cout << "gain_vector : " << filepath << endl;
    }

    return;
}

void dirOriginfile()
{
    string file = "C:/calibration_mentoring/pano/Skull";

    for (const auto& file : directory_iterator(file)) {
        std::string filepath{ file.path().u8string() };
        originfn.push_back(filepath);

        //cout << "origin_vector : " << filepath << endl;
    }
    return;
}