// Condensation_demo.cpp : 定义控制台应用程序的入口点。
 
#include "stdafx.h"
#ifdef _CH_
#pragma package <opencv>
#endif
#include "stdafx.h"
#ifndef _EiC
#include "cv.h"
#include "cvAux.h"
#include "highgui.h"
#include "cxcore.h"
#include <stdio.h>
#include <ctype.h>
#endif
 
 
// 从图片的x、y坐标处返回相应的色调、饱和度和亮度
 
int getpixel(IplImage *image, int x, int y, int *h, int *s, int *v){
*h =(uchar) image->imageData[y *image->widthStep+x * image->nChannels];
*s =(uchar) image->imageData[y *image->widthStep+ x * image->nChannels + 1];
*v =(uchar) image->imageData[y *image->widthStep+ x * image->nChannels + 2];
return 0;
}
//--------------------------------------------------------------------------------
 
 
int main( int argc, char** argv ){
CvCapture* capture = 0;
IplImage* image = 0;
IplImage* HSV = 0;
 
if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0]))){
capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );
}
else if( argc == 2 ){
capture = cvCaptureFromAVI( argv[1] );
}
 
if( !capture ){
fprintf(stderr,"Could not initialize capturing...n");
return -1;
}
 
printf( "Hot keys: \n"
"\tESC - quit the programn");
 
//创建Normal窗口
 
cvNamedWindow("Normal", CV_WINDOW_AUTOSIZE );
 
 
//Condensation结构体初始化-------------------------------------------------
 
int DP=2; // 状态向量的维数
 
int MP=2; // 观测向量的维数
 
int SamplesNum=300; // 样本粒子的数量
 
 
CvConDensation* ConDens=cvCreateConDensation( DP, MP, SamplesNum );
//-----------------------------------------------------------------------
 
 
//Condensation结构体中一些参数的初始化-----------------------------------
 
CvMat* lowerBound; // 下界
 
CvMat* upperBound; // 上界
 
lowerBound = cvCreateMat(2, 1, CV_32F);
upperBound = cvCreateMat(2, 1, CV_32F);
//设置粒子坐标的上下界为窗口大小640*480
 
cvmSet( lowerBound, 0, 0, 0.0 ); cvmSet( upperBound, 0, 0, 640.0 );
cvmSet( lowerBound, 1, 0, 0.0 ); cvmSet( upperBound, 1, 0, 480.0 );
 
cvConDensInitSampleSet(ConDens, lowerBound, upperBound);
//-----------------------------------------------------------------------
 
 
//设置窗口的中心为追踪的初始点------------------------------
 
for(int i=0; i < SamplesNum; i++){
ConDens->flSamples[i][0]+=320.0;
ConDens->flSamples[i][1]+=240.0;
}
//-----------------------------------------------------------------------
 
 
//迁移矩阵的初始化----------------------------
 
ConDens->DynamMatr[0]=1.0;ConDens->DynamMatr[1]=0.0;
ConDens->DynamMatr[2]=0.0;ConDens->DynamMatr[3]=1.0;
//-----------------------------------------------------------------------
 
 
for(;;){
IplImage* frame = 0;
int c;
int X,Y,XX,YY;
int H,S,V;
 
frame = cvQueryFrame( capture );
if( !frame ){
break;
}
 
if( !image ){
image = cvCreateImage( cvGetSize(frame), 8, 3 );
image->origin = frame->origin;
HSV = cvCreateImage( cvGetSize(frame), 8, 3 );
HSV->origin = frame->origin;
}
 
cvCopy( frame, image, 0 );
cvCvtColor(image ,HSV , CV_BGR2HSV);
 
//粒子的置信度计算，置信度需要自己建模---------------------------------------------------
 
for(int i=0; i < SamplesNum; i++){
X=(int)ConDens->flSamples[i][0];
Y=(int)ConDens->flSamples[i][1];
 
if(X>=0 && X<=640 && Y>=0 && Y<=480){ //粒子的坐标在窗口范围之内
 
getpixel(HSV, X, Y, &H, &S, &V);
if(H<=19 && S>=48){ // 肤色的判定 //H<=19 S>=48
 
cvCircle(image, cvPoint(X,Y), 4, CV_RGB(255,0,0), 1);
ConDens->flConfidence[i]=1.0;
}
else{
ConDens->flConfidence[i]=0.0;
}
}
else{
ConDens->flConfidence[i]=0.0;
}
}
//--------------------------------------------------------------------------
 
 
//更新滤波器状态
 
cvConDensUpdateByTime(ConDens);
 
cvShowImage( "Normal", image );
c = cvWaitKey(20);
 
if( c == 27 ){
break;
}
}
 
//释放内存------------------------------------
 
cvReleaseImage(&image);
cvReleaseImage(&HSV);
cvReleaseConDensation(&ConDens);
cvReleaseMat( &lowerBound );
cvReleaseMat( &upperBound );
cvReleaseCapture( &capture );
cvDestroyWindow("Normal");
//---------------------------------------------
 
 
return 0;
}
 
#ifdef _EiC
main(1,"condensation.cpp");
#endif