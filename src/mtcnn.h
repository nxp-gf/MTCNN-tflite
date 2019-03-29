#ifndef MTCNN_H
#define MTCNN_H
//#include "network.h"
#include "opencv2/opencv.hpp"
//#include <algorithm>
#include <stdlib.h>
//#include <memory.h>
#include <fstream>
#include <cstring>
#include <string>
#include <math.h>
#include "pBox.h"

#include "FaceInfo.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#define IMAGE_HEIGHT 480
#define IMAGE_WIDTH  640
using namespace cv;
static string modeldir = "tensorflow/lite/examples/label_image/MTCNN-light/model";

class Pnet
{
public:
    Pnet();
    ~Pnet();
    void run(Mat &image);

    float nms_threshold;
    mydataFmt Pthreshold;
    bool firstFlag;
    vector<struct Bbox> boundingBox_;
    vector<orderScore> bboxScore_;

    int init(const std::string modeldir, float inscale);
private:
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;

    int outw;
    int outh;
    float scale;
    float *outscore;
    float *outlocation;

    void generateBbox();
};

class Rnet
{
public:
    Rnet();
    ~Rnet();
    float Rthreshold;
    void run(Mat &image);
    float score;
    float location[4];
private:
    int outnum = 3;

    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;
};

class Onet
{
public:
    Onet();
    ~Onet();
    void run(Mat &image);
    float Othreshold;
    float score;
    float location[4];
    float keyPoint[10];
private:
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;
};

class mtcnn
{
public:
    mtcnn();
    ~mtcnn();
    int init(std::string modeldir="model");
    FaceDetectionResult detect(const Mat &image,std::vector<FaceInfo>&faces);
    int release();
private:
    const int minsize = 40;
    Mat reImage;
    float nms_threshold[3];
    vector<float> scales_;
    Pnet *simpleFace_;
    vector<struct Bbox> firstBbox_;
    vector<struct orderScore> firstOrderScore_;
    Rnet refineNet;
    vector<struct Bbox> secondBbox_;
    vector<struct orderScore> secondBboxScore_;
    Onet outNet;
    vector<struct Bbox> thirdBbox_;
    vector<struct orderScore> thirdBboxScore_;
};

void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname = "Union");
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width);

static bool cmpScore(struct orderScore lsh, struct orderScore rsh){
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}

#endif
