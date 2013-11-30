/*
使用方法：


1.创建并初始化

const int stateNum=4;//状态数
const int measureNum=2;//测量变量数
const int sampleNum=2000;//粒子数

 CvConDensation* condens = cvCreateConDensation(stateNum,measureNum,sampleNum);

在不影响性能的情况下，粒子数量越大，系统表现的越稳定

其他初始化内容请参考learning opencv


2.预测
3.更新例子可信度，也就是权重。本例中更新方法与learning opencv中有所不同，想看代码 
4.更新CvConDensation
*/

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cvaux.h>
 
#include <cmath>
#include <vector>
#include <iostream>
using namespace std;
 
const int winHeight=600;
const int winWidth=800;
 
 
CvPoint mousePosition=cvPoint(winWidth>>1,winHeight>>1);
 
//mouse event callback
void mouseEvent(int event,int x,int y,int flags,void *param )
{
    if (event==CV_EVENT_MOUSEMOVE) {
        mousePosition=cvPoint(x,y);
    }
}
 
int main (void)
{
    //1.condensation setup
    const int stateNum=4;
    const int measureNum=2;
    const int sampleNum=2000;
 
    CvConDensation* condens = cvCreateConDensation(stateNum,measureNum,sampleNum);
    CvMat* lowerBound;
    CvMat* upperBound;
    lowerBound = cvCreateMat(stateNum, 1, CV_32F);
    upperBound = cvCreateMat(stateNum, 1, CV_32F);
    cvmSet(lowerBound,0,0,0.0 ); 
    cvmSet(upperBound,0,0,winWidth );
    cvmSet(lowerBound,1,0,0.0 ); 
    cvmSet(upperBound,1,0,winHeight );
    cvmSet(lowerBound,2,0,0.0 ); 
    cvmSet(upperBound,2,0,0.0 );
    cvmSet(lowerBound,3,0,0.0 ); 
    cvmSet(upperBound,3,0,0.0 );
    float A[stateNum][stateNum] ={
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1
    };
    memcpy(condens->DynamMatr,A,sizeof(A));
    cvConDensInitSampleSet(condens, lowerBound, upperBound);
 
    CvRNG rng_state = cvRNG(0xffffffff);
    for(int i=0; i < sampleNum; i++){
        condens->flSamples[i][0] = float(cvRandInt( &rng_state ) % winWidth); //width
        condens->flSamples[i][1] = float(cvRandInt( &rng_state ) % winHeight);//height
    }
 
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SCRIPT_COMPLEX,1,1);
 
    char* winName="condensation";
    cvNamedWindow(winName);
    cvSetMouseCallback(winName,mouseEvent);
    IplImage* img=cvCreateImage(cvSize(winWidth,winHeight),8,3);
    bool isPredictOnly=false;//trigger for prediction only,press SPACEBAR
    while (1){
        //2.condensation prediction
        CvPoint predict_pt=cvPoint((int)condens->State[0],(int)condens->State[1]);
 
        float variance[measureNum]={0};     
        //get variance/standard deviation of each state
        for (int i=0;i<measureNum;i++) {
            //sum
            float sumState=0;
            for (int j=0;j<condens->SamplesNum;j++) {
                sumState+=condens->flSamples[i][j];
            }
            //average
            sumState/=sampleNum;
            //variance
            for (int j=0;j<condens->SamplesNum;j++) {
                variance[i]+=(condens->flSamples[i][j]-sumState)*
                    (condens->flSamples[i][j]-sumState);
            }
            variance[i]/=sampleNum-1;
        }
        //3.update particals confidence
        CvPoint pt;
        if (isPredictOnly) {
            pt=predict_pt;
        }else{
            pt=mousePosition;
        }
        for (int i=0;i<condens->SamplesNum;i++) {
            float probX=(float)exp(-1*(pt.x-condens->flSamples[i][0])
                *(pt.x-condens->flSamples[i][0])/(2*variance[0]));
            float probY=(float)exp(-1*(pt.y-condens->flSamples[i][1])
                *(pt.y-condens->flSamples[i][1])/(2*variance[1]));
            condens->flConfidence[i]=probX*probY;
        }
        //4.update condensation
        cvConDensUpdateByTime(condens);
         
        //draw 
        cvSet(img,cvScalar(255,255,255,0));
        cvCircle(img,predict_pt,5,CV_RGB(0,255,0),3);//predicted point with green
        char buf[256];
        sprintf_s(buf,256,"predicted position:(%3d,%3d)",predict_pt.x,predict_pt.y);
        cvPutText(img,buf,cvPoint(10,30),&font,CV_RGB(0,0,0));
        if (!isPredictOnly) {
            cvCircle(img,mousePosition,5,CV_RGB(255,0,0),3);//current position with red
            sprintf_s(buf,256,"real position :(%3d,%3d)",mousePosition.x,mousePosition.y);
            cvPutText(img,buf,cvPoint(10,60),&font,CV_RGB(0,0,0));
        }
         
        cvShowImage(winName, img);
        int key=cvWaitKey(30);
        if (key==27){//esc   
            break;
        }else if (key==' ') {//trigger for prediction
            //isPredict=!isPredict;
            if (isPredictOnly) {
                isPredictOnly=false;
            }else{
                isPredictOnly=true;
            }
        }
    }      
 
    cvReleaseImage(&img);
    cvReleaseConDensation(&condens);
    return 0;
}