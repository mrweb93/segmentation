#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;


Mat FloodFill(Mat image)
{
    Point startPoint;
    startPoint.x = image.cols / 2;
    startPoint.y = image.rows / 2;
    Scalar loDiff(20, 20, 255);
    Scalar upDiff(5, 5, 255);
    Scalar fillColor(0, 0, 255);
    int neighbors = 8;
    Rect domain;
    int area = floodFill(image, startPoint, fillColor, &domain, loDiff, upDiff, neighbors);
    rectangle(image, domain, Scalar(255, 0, 0));

    return image;
}


Mat MeanShift(Mat image)
{
    Mat imageSegment;
    int spatialRadius = 35;
    int colorRadius = 60;
    int pyramidLevels = 3;
    pyrMeanShiftFiltering(image, imageSegment, spatialRadius, colorRadius, pyramidLevels);

    return imageSegment;
}


Mat WaterShed (Mat image)
{
    Mat imageGray, imageBin;
    cvtColor(image, imageGray, CV_BGR2GRAY);
    threshold(imageGray, imageBin, 100, 255, THRESH_BINARY);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(imageBin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    Mat markers(image.size(), CV_32SC1);
    markers = Scalar::all(0);
    int compCount = 0;
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0], compCount++)
    {
        drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
    }
    std::vector<Vec3b> colorTab(compCount);
    for(int i = 0; i < compCount; i++)
    {
        colorTab[i] = Vec3b(rand()&255, rand()&255, rand()&255);
    }
    watershed(image, markers);
    Mat wshed(markers.size(), CV_8UC3);
    for(int i = 0; i < markers.rows; i++)
    {
        for(int j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i, j);
            if(index == -1)  wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else if (index == 0) wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            else  wshed.at<Vec3b>(i, j) = colorTab[index - 1];
        }
    }
    return wshed;
}

Mat Haar(Mat img)
{

    string cascadeHead = "cascade/people_head.xml";
    string cascadeName = "cascade/people.xml";

    CascadeClassifier detectorBody;
    bool loaded1 = detectorBody.load(cascadeName);
    CascadeClassifier detectorHead;
    bool loaded2 = detectorHead.load(cascadeHead);


    Mat original;

    img.copyTo(original);

    vector<Rect> human;
    vector<Rect> head;

    cvtColor(img, img, CV_BGR2GRAY);

    equalizeHist(img, img);

    detectorBody.detectMultiScale(img, human, 1.04, 4, 0 | 1, Size(30, 80), Size(80,200));
    detectorHead.detectMultiScale(img, head, 1.1, 4, 0 | 1, Size(40, 40), Size(100, 100));


    if (human.size() > 0) {
        for (int gg = 0; gg < human.size(); gg++) {

            rectangle(original, human[gg].tl(), human[gg].br(), Scalar(0, 0, 255), 2, 8, 0);

        }
    }

    if (head.size() > 0) {
        for (int gg = 0; gg < head.size(); gg++) {

            rectangle(original, head[gg].tl(), head[gg].br(), Scalar(0, 0, 255), 2, 8, 0);

        }
    }
    return original;
}

void CreateImg(string way, int key, int key_way )
{
    string new_filename,filename;

    for (int i=0; i<100; i++)
    {
        switch (key_way)
        {
            case 1: filename="img/person_"+to_string(i)+".png"; break;
            case 2: filename="ws_img/ws_person_"+to_string(i)+".png"; break;
            case 3: filename="ms_img/ms_person_"+to_string(i)+".png"; break;
            case 4: filename="ff_img/ff_person_"+to_string(i)+".png"; break;
        }

        Mat img_mat = imread(filename.c_str(),1);

        switch (key)
        {
            case 1: img_mat=WaterShed(img_mat); break;
            case 2: img_mat=MeanShift(img_mat); break;
            case 3: img_mat=FloodFill(img_mat); break;
            case 4: img_mat=Haar(img_mat); break;
        }

        new_filename=way+to_string(i)+".png";
        imwrite(new_filename.c_str() ,img_mat);
    }
}

void ShowImg(string way, string nameWin)
{
    int c=0;
    string filename=way+to_string(c)+".png";
    IplImage* img = cvLoadImage(filename.c_str(),1);
    int dstWidth=img->width*20;
    int dstHeight=img->height*5;
    IplImage* dst=cvCreateImage(cvSize(dstWidth,dstHeight),IPL_DEPTH_8U,3);

    for (int i=0; i<20; i++)
    {
        for (int j=0; j<5; j++)
        {
            filename.clear();
            filename=way+to_string(c)+".png";

            img = cvLoadImage(filename.c_str(), 1);
            cvSetImageROI(dst, cvRect(0 + i*img->width, 0+j*img->height, img->width, img->height));
            cvCopy(img, dst, NULL);
            cvResetImageROI(dst);

            c++;
        }
    }
    cvNamedWindow(nameWin.c_str(), CV_WINDOW_AUTOSIZE);
    cvShowImage(nameWin.c_str(), dst );
}

int main()
{
    ShowImg("img/person_","original");

    CreateImg("haar_img/haar_person_",4,1);
    ShowImg("haar_img/haar_person_","Haar+original");

    /////////////////////////////////////////////////////////////

    CreateImg("ws_img/ws_person_",1,1);
    ShowImg("ws_img/ws_person_","WaterShed");

    CreateImg("haar_ws_img/haar_ws_person_",4,2);
    ShowImg("haar_ws_img/haar_ws_person_","Haar+WaterShed");

    /////////////////////////////////////////////////////////////

    CreateImg("ms_img/ms_person_",2,1);
    ShowImg("ms_img/ms_person_","MeanShift");

    CreateImg("haar_ms_img/haar_ms_person_",4,3);
    ShowImg("haar_ms_img/haar_ms_person_","Haar+MeanShift");

    ////////////////////////////////////////////////////////////

     CreateImg("ff_img/ff_person_",3,1);
     ShowImg("ff_img/ff_person_","FloodFill");

    CreateImg("haar_ff_img/haar_ff_person_",4,4);
    ShowImg("haar_ff_img/haar_ff_person_","Haar+FloodFill");


    cvWaitKey(0);

}
