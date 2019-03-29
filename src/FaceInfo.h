#pragma once
#include <stdint.h>
//namespace facedetecion
//{
    enum FaceDetectionResult{
        FaceDetectionResult_OK,
        MODELS_NOT_FOUND,
        TOO_LESS_BBOXES_PNET,
        TOO_LESS_BBOXES_RNET,
        TOO_LESS_BBOXES_ONET,
    };
    typedef struct Rect_tag {
        int x;
        int y;
        int width;
        int height;
    } Rect;
    typedef struct FaceInfo_tag {
        struct Rect_tag bbox;
        double roll;
        double pitch;
        double yaw;
        double score;
    } FaceInfo;
//}