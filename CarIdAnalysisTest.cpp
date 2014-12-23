// CarIdAnalysisTest.cpp : 定义控制台应用程序的入口点。
//
//参考文献：郑成勇，一种RGB颜色空间中的车牌定位新方法，中国图象图形学报，2010年11 月，15卷，第11期。
#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <assert.h>
#include <list>
#include <stack>
#include <iostream>
using namespace std;

#define Debug 1
//调试用代码，显示图像
void _dShowImage(const IplImage* pImg)
{
#ifdef Debug
	const char* winName = "debug";
	cvNamedWindow(winName,CV_WINDOW_AUTOSIZE);
	cvShowImage(winName,pImg);
	cvWaitKey(0);
	cvDestroyWindow(winName);
#endif
}

typedef unsigned char uchar;
//颜色提取
void PickupBlueColor(const IplImage* src,IplImage* dst,double* m,double* p)
{
	const double k1 = 0.33748,k2= 0.66252;
	double d1,d2;
	long N = 0,M=0,P=0;
	for (int r=0;r<src->height;r++)
	{
		for (int c=0;c<src->width;c++)
		{
			CvScalar color = cvGet2D(src,r,c);
			d1 = color.val[0] - color.val[2] ;
			d2 = color.val[0] - color.val[1];
			uchar gray = d1 <= 0 ? 0 : d2<= 0 ? 0 :(uchar)(k1*d1 + k2*d2);
			N += gray > 0 ? 1 : 0;
			M += gray ;
			CV_IMAGE_ELEM(dst,uchar,r,c) = gray;
		}
	}
	*m = (double)M/N;

	for (int r=0;r<src->height;r++)
	{
		for (int c=0;c<src->width;c++)
		{
			P += CV_IMAGE_ELEM(dst,uchar,r,c) > *m ? 1 : 0;
		}
	}
	*p = (double)P/N;
}

void ThreshOld(const IplImage* src,IplImage* dst,double m,double p)
{
	uchar thr = (uchar)(m*( 0.7 + 0.3 /p));
	for (int r=0;r<src->height;r++)
	{
		for (int c=0;c<src->width;c++)
		{
			CV_IMAGE_ELEM(dst,uchar,r,c) = CV_IMAGE_ELEM(src,uchar,r,c) > thr ? 255 : 0;
		}
	}
}
//绘制目标
void DrawBox(IplImage* src,CvBox2D box)
{
	CvPoint2D32f point[4];
	int i;
	for ( i=0; i<4; i++)
	{
		point[i].x = 0;
		point[i].y = 0;
	}
	cvBoxPoints(box, point); //计算二维盒子顶点
	CvPoint pt[4];
	for ( i=0; i<4; i++)
	{
		pt[i].x = (int)point[i].x;
		pt[i].y = (int)point[i].y;
	}
	cvLine( src, pt[0], pt[1],CV_RGB(255,0,0), 2, 8, 0 );
	cvLine( src, pt[1], pt[2],CV_RGB(255,0,0), 2, 8, 0 );
	cvLine( src, pt[2], pt[3],CV_RGB(255,0,0), 2, 8, 0 );
	cvLine( src, pt[3], pt[0],CV_RGB(255,0,0), 2, 8, 0 );
}
//形态学操作
void Morphology(const IplImage* src,IplImage* dst)
{
	IplImage* f1 = cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	IplImage* f2 = cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	IplImage* tmp = cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);

	IplConvKernel* k225 = cvCreateStructuringElementEx(3,25, 1, 12, CV_SHAPE_RECT);
	IplConvKernel* k210 = cvCreateStructuringElementEx(3,11, 1, 5, CV_SHAPE_RECT);
	IplConvKernel* k102 = cvCreateStructuringElementEx(11,3, 5, 1, CV_SHAPE_RECT);
	
	cvMorphologyEx(src, f1, tmp,k225, CV_MOP_CLOSE, 1);
	cvMorphologyEx(f1, f2, tmp,k102, CV_MOP_OPEN,  1);
	cvMorphologyEx(f2, dst, tmp,k210, CV_MOP_OPEN,  1);

	cvReleaseStructuringElement(&k225);
	cvReleaseStructuringElement(&k210);
	cvReleaseStructuringElement(&k102);
	cvReleaseImage(&f1);
	cvReleaseImage(&f2);
	cvReleaseImage(&tmp);
}
//计算两个相邻Box的最小外接矩形
void BoxsMinAreaRect(const CvBox2D box1,const CvBox2D box2,CvBox2D &box3)
{
	CvPoint2D32f points1[4],points2[4];
	CvMat* vector = cvCreateMat( 1, 8, CV_32SC2 );
	memset(points1,0,4*sizeof(CvPoint2D32f));
	memset(points2,0,4*sizeof(CvPoint2D32f));

	cvBoxPoints(box1, points1); 
	cvBoxPoints(box2, points2); 
	
	for (int i=0;i<4;i++)
	{
		CV_MAT_ELEM( *vector, CvPoint, 0, i ) = cvPoint((int)points1[i].x,(int)points1[i].y);
		CV_MAT_ELEM( *vector, CvPoint, 0, 4+i ) = cvPoint((int)points2[i].x,(int)points2[i].y);
	}
	box3 = cvMinAreaRect2(vector);
	cvReleaseMat(&vector);
}
//相邻Box合并
void MergeRoi(IplImage* src,list<CvBox2D> &lstBox)
{
	lstBox.clear();
	CvMemStorage* strorage = cvCreateMemStorage();
	CvSeq* pSeqHead,*pSeqTmp = NULL;
	int num = cvFindContours(src,strorage,&pSeqHead);
	const float MinArea = 400;
	const float MaxArea = 100000;
	list<CvBox2D>::iterator it;

	for (pSeqTmp=pSeqHead;  pSeqTmp;  pSeqTmp=pSeqTmp->h_next)
	{
		CvBox2D box = cvMinAreaRect2(pSeqTmp);
		float boxArea = box.size.width*box.size.width;
		if( boxArea < MinArea || boxArea > MaxArea)
			continue;
		if(lstBox.empty())
		{
			lstBox.push_back(box);
			continue;
		}
		//DrawBox(colorScale,box);
		CvBox2D box2;
		box2.center.x = 0;
		for (it=lstBox.begin();it!=lstBox.end();it++)
		{
			if(it->angle - box.angle <= 1)
			{
				float disx = fabs(fabs(it->center.x - box.center.x) - (it->size.width+box.size.width)/2);
				float disy = fabs(fabs(it->center.y - box.center.y) - (it->size.height+box.size.height)/2);
				if(disx < 50)
				{
					BoxsMinAreaRect(*it,box,box2);
					it->center = box2.center;
					it->size = box2.size;
				}
			}	
		}
		if(!box2.center.x)
			lstBox.push_back(box);
	}
	cvReleaseMemStorage(&strorage);
}
int _tmain(int argc, _TCHAR* argv[])
{
	const char* strImgSrcPath = "E:\\在研项目\\车牌抠图\\测试集\\3.jpg";
	IplImage* colorScale = cvLoadImage(strImgSrcPath,CV_LOAD_IMAGE_COLOR);
	IplImage* f = cvCreateImage(cvGetSize(colorScale),IPL_DEPTH_8U,1);
	IplImage* f0 = cvCreateImage(cvGetSize(f),IPL_DEPTH_8U,1);
	double m,p;
	PickupBlueColor(colorScale,f,&m,&p);
	//_dShowImage(f);
	ThreshOld(f,f0,m,p);
	//_dShowImage(f0);
	IplImage* f3 = cvCreateImage(cvGetSize(f),IPL_DEPTH_8U,1);
	Morphology(f0,f3);
	//_dShowImage(f3);
	list<CvBox2D> lstBox;
	MergeRoi(f3,lstBox);
	list<CvBox2D>::iterator it;
	float minrate = 1;
	CvBox2D box;
	IplImage* grayScale = cvCreateImage(cvGetSize(colorScale),IPL_DEPTH_8U,1);
	cvCvtColor(colorScale,grayScale,CV_BGR2GRAY);

	for (it=lstBox.begin();it!=lstBox.end();it++)
	{
		CvPoint2D32f points[4];
		cvBoxPoints(*it,points);
		int minx,maxx,miny,maxy;
		minx = maxx = (int)points[0].x;
		miny = maxy = (int)points[0].y;
		for (int i=0;i<4;i++)
		{
			minx = minx > (int)points[i].x ? (int)points[i].x : minx;
			miny = miny > (int)points[i].y ? (int)points[i].y : miny;
			maxx = maxx < (int)points[i].x ? (int)points[i].x : maxx;
			maxy = maxy < (int)points[i].x ? (int)points[i].y : maxy;
		}
		CvRect roi = cvRect(minx,miny,maxx-minx,maxy-miny);
		IplImage* roiImg = cvCreateImage(cvSize(roi.width,roi.height),grayScale->depth,grayScale->nChannels);
		cvSetImageROI(grayScale,roi);
		cvCopy(grayScale,roiImg);
		cvResetImageROI(colorScale);
		cvThreshold(roiImg,roiImg,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
		int frontsum(0),backsum(0);
		for (int r=0;r<roi.height;r++)
			for (int c=0;c<roi.width;c++)
				CV_IMAGE_ELEM(roiImg,uchar,r,c) == 255 ? frontsum++ : backsum++;
		double fbrate = (double)frontsum/backsum;
		if(fbrate > 0.7 || fbrate < 0.3)
			continue;
		_dShowImage(roiImg);
		cvReleaseImage(&roiImg);
	}
	_dShowImage(colorScale);

	return 0;
}