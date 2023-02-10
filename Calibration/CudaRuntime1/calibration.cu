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

//char path1[1000];
//char width1[10];
//char height1[10];

void dirDarkfile();      //              Dark    丮        raw       darkfn    Ϳ  pushback  ϴ   Լ  
void dirGainfile();      //              Gain    丮        raw       gainfn    Ϳ  pushback  ϴ   Լ  
void darkMap();         // darkfn     հ      darkMap.raw       Լ
void gainMap();         // gainfn     հ      darkMap.raw       Լ

vector<string> darkfn;   // Dark    丮    raw   ϵ         ϰ   ִ      
vector<string> gainfn;   // Gain    丮    raw   ϵ         ϰ   ִ      
vector<string> originfn;

void dirOriginfile();
void callibration();   //       Լ


__global__ void cudaFilesSum(ushort* inimg, float* averageimg, int width, int height)
{
   // printf("width : %d, height : %d", width, height);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + (y * width);

    if (x < width && y < height) {
        *(averageimg + offset) += *(inimg + offset);
    }
}

__global__ void cudaPixelAvg(float* averageimg, ushort* outimg, int width, int height)
{
    //rintf("width : %d, height : %d", width, height);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + (y * width);

    if (x < width && y < height) {
        *(outimg + offset) = *(averageimg + offset) / 1742;
    }
}

__global__ void cudaCalibration(ushort* darkMapImg, ushort* GainMapImg,
    ushort* MTF_VImg, double subGainAvg,
    int width, int height, ushort* outimg)
{
    //printf("width : %d, height : %d", width, height);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + (y * width);

    if (x < width && y < height) {
        *(outimg + offset) =
            (ushort)(abs(*(MTF_VImg + offset) - (*(darkMapImg + offset))) / (float)((*(GainMapImg + offset)) - (*(darkMapImg + offset))) * subGainAvg);
            //(ushort)(abs(*(MTF_VImg + offset) - (*(darkMapImg + offset))));
    }
}

int main(int argc, char* argv[])
{
    //memcpy(path1, argv[1], 1000);
    //memcpy(width1, argv[2], sizeof(argv[2]));
    //memcpy(height1, argv[3], sizeof(argv[3]));

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

    //printf("%s, %s, %s\n", path1, width1, height1);
    printf("Success processing !!\n");

    return 0;
}

void darkMap() {
    FILE* infp, * outfp;
    //char savefile[] = "C:\\m_data_230110\\S1_1628x1628\\DarkMap(1628).raw";
    //string pathStr(path1);
    string file = "C:/calibration_mentoring/pano/DarkMap(pano).raw";
    //char savefile[] = file.c_str();

    ushort* inimg, * outimg;
    float* f_averageimg;

    //int width = atoi(width1);
    //int height = atoi(height1);
    int width = 80, height = 1628;
    //int width = atoi(width1);
    //int height = atoi(height1);
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
    iter = darkfn.begin();
    for (iter = darkfn.begin(); iter != darkfn.end(); iter++) {
        memset(inimg, 0, sizeof(ushort) * imageSize);
        //char path[100] = "C:\\m_data_230110\\S1_1628x1628\\Dark\\";
        string file2 = *iter;

        //cout << "path: " << file2 << endl;   //      fopen Ȯ

        if ((infp = fopen(file2.c_str(), "rb")) == NULL) {
            //cout << "path : " << file.c_str() << endl;
            printf("%d No such file or folder\n", __LINE__);
            return;
        }

        fread(inimg, sizeof(ushort) * imageSize, 1, infp);

        /* cuda reset */
        ushort* d_inimg = NULL;

        cudaMalloc(&d_inimg, sizeof(ushort) * imageSize);
        cudaMemset(d_inimg, 0, sizeof(ushort) * imageSize);
        cudaMemcpy(d_inimg, inimg, sizeof(ushort) * imageSize, cudaMemcpyHostToDevice);


        cudaFilesSum << <dimGrid, dimBlock >> > (d_inimg, d_averageimg, width, height);
        cudaFree(d_inimg);

        fclose(infp);
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


void gainMap() {
    FILE* infp, * outfp;
    //char savefile[] = "C:\\m_data_230110\\S1_1628x1628\\GainMap(1628).raw";
    //string pathStr(path1);
    string file = "C:/calibration_mentoring/pano/GainMap(pano).raw";
    // char savefile[] = file.c_str();

    ushort* inimg, * outimg;
    float* f_averageimg;

    //int width = atoi(width1);
    //int height = atoi(height1);
    int width = 80, height = 1628;
    int imageSize = width * height;

    //cout << "imagesize : " << imageSize << endl;

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
        //char path[100] = "C:\\m_data_230110\\S1_1628x1628\\Gain\\";
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
    //char savefile[] = "C:\\m_data_230110\\S1_1628x1628\\callibration(1628)2.raw";
    //char savefile[] = file.c_str();
    // 
    //int width = atoi(width1);
    //int height = atoi(height1);
    int width = 80, height = 1628;
    int imageSize = width * height;
    //int subImageSize = (width - 200) * (height - 200);
    int subImageSize = width * height;

//    int widthcnt = 0;
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
        //char savefile[] = "C:\\m_data_230110\\S1_1628x1628\\GainMap(1628).raw";
        //string file2 = pathStr + "/GainMap.raw";
        //char savefile[] = file2.c_str();


        //string dark_path = pathStr + "/DarkMap.raw";
        //string gain_path = pathStr + "/GainMap.raw";
        //string origin_path = pathStr + "/MTF_V.raw";
        if ((MTF_VFp = fopen(origin_file.c_str(), "rb")) == NULL) {
            printf("%d No such file or folder\n", __LINE__);
            return;
        }
        cout << "origin file : " << * iter << endl;

        fread(MTF_VImg, sizeof(ushort) * imageSize, 1, MTF_VFp);

        fclose(MTF_VFp);

        // const dim3 dimGrid((int)ceil((width / 4)), (int)ceil((height) / 4));
         //const dim3 dimBlock(4, 4);

        cudaMalloc(&d_MTF_VImg, sizeof(ushort) * imageSize);
        cudaMemset(d_MTF_VImg, 0, sizeof(ushort) * imageSize);
        cudaMemcpy(d_MTF_VImg, MTF_VImg, sizeof(ushort) * imageSize, cudaMemcpyHostToDevice);

        cudaMalloc(&d_outimg, sizeof(ushort) * imageSize);
        cudaMemset(d_outimg, 0, sizeof(ushort) * imageSize);


        subGainAvg = 0;
        subGainSum = 0;

        for (int i = 0; i < imageSize; i++) {
            //widthcnt++;
            //if (widthcnt == width) widthcnt = 0;

            //if (width * 100 > i || width * (height - 100) < i) continue;
            //if (widthcnt <= 100 || widthcnt > 1528) continue;

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

        string file = "C:/calibration_mentoring/exe (Pano)/frame/ " + fileName;

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
    //char savefile[] = "C:\\m_data_230110\\S1_1628x1628\\callibration(1628)2.raw";

//    string pathStr(path1);
    string file = "C:/calibration_mentoring/pano/Dark";
    //char path[] = file.c_str();

    for (const auto& file : directory_iterator(file)) {
        std::string filepath{ file.path().u8string() };
        darkfn.push_back(filepath);
    }

    return;
}

void dirGainfile()
{
    //string path = "C:\\m_data_230110\\S1_1628x1628\\Gain\\";

    string file = "C:/calibration_mentoring/pano/Gain_No_Filter/70kV_5mA";
    //char path[] = file.c_str();

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
    //char path[] = file.c_str();

    for (const auto& file : directory_iterator(file)) {
        std::string filepath{ file.path().u8string() };
        originfn.push_back(filepath);

        //cout << "origin_vector : " << filepath << endl;
    }
    return;
}