#ifndef PBOX_H
#define PBOX_H
#include <stdlib.h>
#include <iostream>

using namespace std;
#define mydataFmt float

class pBox
{
public:
    pBox(){pdata = NULL;}
    ~pBox(){
        if(pdata) {
            delete pdata;
            pdata = NULL;
        }
    }

    mydataFmt *pdata;
    int width;
    int height;
    int channel;
};

class pRelu
{
public:
    pRelu(){pdata = NULL;}
    ~pRelu() {
        if(pdata) {
            delete pdata;
            pdata = NULL;
        }
    }
    mydataFmt *pdata;
    int width;
};

class Weight
{
public:
    Weight(){pdata = NULL; pbias = NULL;}
    ~Weight() {
        if(pdata) {
            delete pdata;
            pdata = NULL;
        }
        if(pbias) {
            delete pbias;
            pbias = NULL;
        }
    }
    mydataFmt *pdata;
    mydataFmt *pbias;
    int lastChannel;
    int selfChannel;
    int kernelSize;
    int stride;
    int pad;
};

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    mydataFmt ppoint[10];
    mydataFmt regreCoord[4];
};

struct orderScore
{
    mydataFmt score;
    int oriOrder;
};

void freepBox(pBox *pbox);
void freeWeight(Weight *weight);
void freepRelu(pRelu *prelu);
void pBoxShow(const pBox *pbox);
void pBoxShowE(const pBox *pbox,int channel, int row);
void weightShow(const Weight *weight);
void pReluShow(const pRelu *prelu);
#endif
