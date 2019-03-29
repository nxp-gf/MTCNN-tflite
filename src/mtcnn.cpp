#include "mtcnn.h"
#include "stdio.h"
#include "math.h"

#define LOG(x) std::cerr
Pnet::Pnet(){
    Pthreshold = 0.9;
    nms_threshold = 0.5;
}

string pnet_models[] = {"pnet_144_192.tflite", 
                        "pnet_103_137.tflite",
                        "pnet_73_97.tflite",
                        "pnet_52_69.tflite",
                        "pnet_37_49.tflite",
                        "pnet_26_35.tflite",
                        "pnet_19_25.tflite",
                        "pnet_13_18.tflite"};

int Pnet::init(const string model_path, float inscale){
    // 1. create model
    //std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model\n";
        return -1;
    }
    LOG(INFO) << "Loaded model\n";
    model->error_reporter();
    LOG(INFO) << "resolved reporter\n";

    // 2. create OpResolver
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // 3. create Interpreter
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        return -1;
    }

    // 4. setting parameter for Interpreter
    interpreter->UseNNAPI(0); // using NNAPI for accel?
    interpreter->SetAllowFp16PrecisionForFp32(0); // set date format
    interpreter->SetNumThreads(1); // set running threads number

    // 5. alloc inputs/outputs tensor memory
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    scale = inscale;
    outw = (int)ceil(IMAGE_WIDTH * scale * 0.5 - 5);
    outh = (int)ceil(IMAGE_HEIGHT * scale * 0.5 - 5);
    outscore = new float[outw * outh * 2];
    outlocation = new float[outw * outh * 4];
    return 0;
}
Pnet::~Pnet(){
    /*Release Pnet*/
}

void image2Matrix(const Mat &image, float *p){
    if ((image.data == NULL) || (image.type() != CV_8UC3)){
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    for (int rowI = 0; rowI < image.rows; rowI++){
        for (int colK = 0; colK < image.cols; colK++){
            *p = (image.at<Vec3b>(rowI, colK)[0] - 127.5)*0.0078125;
            *(p + 1) = (image.at<Vec3b>(rowI, colK)[1] - 127.5)*0.0078125;
            *(p + 2) = (image.at<Vec3b>(rowI, colK)[2] - 127.5)*0.0078125;
            p += 3;
        }
    }
}


void Pnet::run(Mat &image){
    // 12. start Invoke
    int input = interpreter->inputs()[0];
    image2Matrix(image, interpreter->typed_tensor<float>(input));
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    float *scores = interpreter->typed_output_tensor<float>(0);
    float *location = interpreter->typed_output_tensor<float>(1);
    for (int i = 0; i < outh * outw; ++i) {
      outscore[i] = scores[i * 2 + 1];
    }
    for (int i = 0; i < outh * outw * 4; ++i) {
      outlocation[i] = location[i];
    }

    /*Run Rnet*/
    generateBbox();
}
void Pnet::generateBbox(){
    //for pooling 
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    mydataFmt *p = outscore;
    mydataFmt *plocal = outlocation;
    struct Bbox bbox;
    struct orderScore order;
    for(int row = 0; row < outh; row ++){
        for(int col = 0; col < outw; col ++){
            if(*p > Pthreshold){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*row+1)/scale);
                bbox.y1 = round((stride*col+1)/scale);
                bbox.x2 = round((stride*row+1+cellsize)/scale);
                bbox.y2 = round((stride*col+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=*(plocal + channel);
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal += 4;
        }
    }
}

Rnet::Rnet(){
    Rthreshold = 0.7;
    /* Init Rnet*/

    // 1. create model
    //std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile("model/rnet.tflite");
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model\n";
        //return -1;
    }
    LOG(INFO) << "Loaded model\n";
    model->error_reporter();
    LOG(INFO) << "resolved reporter\n";

    // 2. create OpResolver
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // 3. create Interpreter
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        //return -1;
    }

    // 4. setting parameter for Interpreter
    interpreter->UseNNAPI(0); // using NNAPI for accel?
    interpreter->SetAllowFp16PrecisionForFp32(0); // set date format
    interpreter->SetNumThreads(1); // set running threads number

    // 5. alloc inputs/outputs tensor memory
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }
}
Rnet::~Rnet(){
    /* Release Rnet*/
}
void Rnet::run(Mat &image){
    // 12. start Invoke
    int input = interpreter->inputs()[0];
    image2Matrix(image, interpreter->typed_tensor<float>(input));
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    float *out0 = interpreter->typed_output_tensor<float>(0);
    float *out1 = interpreter->typed_output_tensor<float>(1);
    score = out0[1];
    for (int i = 0; i < 4; ++i) {
      location[i] = out1[i];
    }
}

Onet::Onet(){
    Othreshold = 0.8;
    /*Init*/

    // 1. create model
    //std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile("model/onet.tflite");
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model\n";
        //return -1;
    }
    LOG(INFO) << "Loaded model\n";
    model->error_reporter();
    LOG(INFO) << "resolved reporter\n";

    // 2. create OpResolver
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // 3. create Interpreter
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        //return -1;
    }

    // 4. setting parameter for Interpreter
    interpreter->UseNNAPI(0); // using NNAPI for accel?
    interpreter->SetAllowFp16PrecisionForFp32(0); // set date format
    interpreter->SetNumThreads(1); // set running threads number

    // 5. alloc inputs/outputs tensor memory
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

}
Onet::~Onet(){
    /*Release*/
}
void Onet::run(Mat &image){
    // 12. start Invoke
    int input = interpreter->inputs()[0];
    image2Matrix(image, interpreter->typed_tensor<float>(input));
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    float *out0 = interpreter->typed_output_tensor<float>(0);
    float *out1 = interpreter->typed_output_tensor<float>(1);
    float *out2 = interpreter->typed_output_tensor<float>(2);

    score = out0[1];
    for (int i = 0; i < 4; ++i) {
      location[i] = out1[i];
    }
    for (int i = 0; i < 10; ++i) {
      keyPoint[i] = out2[i];
    }
}


mtcnn::mtcnn(){
    init();
}

mtcnn::~mtcnn(){
    release();
}

int mtcnn::init(std::string modeldir)
{
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;

    float minl = IMAGE_HEIGHT < IMAGE_WIDTH ? IMAGE_HEIGHT : IMAGE_WIDTH;
    int MIN_DET_SIZE = 12;
    float m = (float)MIN_DET_SIZE / minsize;
    minl *= m;
    float factor = 0.709;
    while (minl > MIN_DET_SIZE) {
        scales_.push_back(m);
        minl *= factor;
        m = m*factor;
    }
    simpleFace_ = new Pnet[scales_.size()];

    for (size_t i = 0; i < scales_.size(); i++) {
        simpleFace_[i].init("model/" + pnet_models[i], scales_[i]);
    }
    return 0;
}

int mtcnn::release()
{
    scales_.clear();

    if (simpleFace_) {
    	delete[]simpleFace_;
        simpleFace_ = NULL;
        return 0;
    }
    return -1;
}

enum FaceDetectionResult mtcnn::detect(const Mat &image, std::vector<FaceInfo> &faces){
    faces.clear();
    struct orderScore order;
    int count = 0;

    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
        simpleFace_[i].run(reImage);
        nms(simpleFace_[i].boundingBox_, simpleFace_[i].bboxScore_, simpleFace_[i].nms_threshold);


        vector<struct Bbox>::iterator it;
        for(it=simpleFace_[i].boundingBox_.begin(); it!=simpleFace_[i].boundingBox_.end(); it ++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        simpleFace_[i].bboxScore_.clear();
        simpleFace_[i].boundingBox_.clear();
    }

    //the first stage's nms
    if(count<1)return TOO_LESS_BBOXES_PNET;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols);

    //second stage
    count = 0;
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            cv::Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat secImage;
            resize(image(temp), secImage, Size(24, 24), 0, 0, cv::INTER_LINEAR);
            refineNet.run(secImage);
            if(refineNet.score > refineNet.Rthreshold){
                memcpy(it->regreCoord, refineNet.location, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = refineNet.score;
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    if(count<1)return TOO_LESS_BBOXES_RNET;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols);

    //third stage 
    count = 0;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            cv::Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat thirdImage;
            resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_LINEAR);
            outNet.run(thirdImage);
            mydataFmt *pp=NULL;
            if(outNet.score>outNet.Othreshold){
                memcpy(it->regreCoord, outNet.location, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = outNet.score;
                pp = outNet.keyPoint;
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
                }
                for(int num=0;num<5;num++){
                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
                }
                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else{
                it->exist=false;
            }
        }
    }

    if(count<1)return TOO_LESS_BBOXES_ONET;
    refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    count = 0;
    for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++){
        if((*it).exist){
            FaceInfo fi;
            fi.bbox.x = (*it).x1;
            fi.bbox.y = (*it).y1;
            fi.bbox.width = (*it).x2 - (*it).x1;
            fi.bbox.height = (*it).y2 -(*it).y1;
            faces.push_back(fi);
            cv::rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(255,0,255), 2,8,0);
            for(int num=0;num<5;num++)
                circle(image,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
        }
    }
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
    return FaceDetectionResult_OK;
}

void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for(int num = 0;num < boundingBox_.size();num++){
            if(boundingBox_.at(num).exist){
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min")){
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(unsigned int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<struct Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbh = (*it).x2 - (*it).x1 + 1;
            bbw = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[1]*bbh;
            y1 = (*it).y1 + (*it).regreCoord[0]*bbw;
            x2 = (*it).x2 + (*it).regreCoord[3]*bbh;
            y2 = (*it).y2 + (*it).regreCoord[2]*bbw;

            h = x2 - x1 + 1;
            w = y2 - y1 + 1;

            maxSide = (h>w)?h:w;
            x1 = x1 + h*0.5 - maxSide*0.5;
            y1 = y1 + w*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>height)(*it).x2 = height - 1;
            if((*it).y2>width)(*it).y2 = width - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}

