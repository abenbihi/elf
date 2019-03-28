// test.cpp --- 
// 
// Filename: test.cpp
// Description: 
// Author: Yannick Verdie, Kwang Moo Yi
// Maintainer: Yannick Verdie
// Created: Tue Mar  3 17:47:28 2015 (+0100)
// Version: 0.5a
// Package-Requires: ()
// Last-Updated: Tue Jun 16 17:09:04 2015 (+0200)
//           By: Kwang
//     Update #: 26
// URL: 
// Doc URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change Log:
// 
// 
// 
// 
// Copyright (C), EPFL Computer Vision Lab.
// 
// 

// Code:

#include "src/libTILDE.hpp"
#include <chrono>
#include <opencv/cv.h>  	//for parallel_opencv
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <string>
#include <fstream>
#include <sstream>
//#include <boost/filesystem.hpp>

//#include <utility>      // std::pair

//using namespace cv;

vector<KeyPoint> testAndDump(const Mat &I,const string &pathFilter, const int &nbTest = 1, const char* ext = NULL, Mat* score = NULL)
{
	using namespace std::chrono;
	using namespace cv;
 	high_resolution_clock::time_point t1, t2;
 	std::vector<KeyPoint> kps;
 	double time_spent = 0;

	// Use appoximated filters if told to do so
 	bool useApprox = false;
 	if (ext != NULL)
 		useApprox = true;

	// Run multiple times to measure average runtime
	for (int i =0;i<nbTest;i++)
	{
		t1 = high_resolution_clock::now();
		// Run TILDE
	    kps = getTILDEKeyPoints(I, pathFilter, useApprox,true,-std::numeric_limits<float>::infinity(),score);
		t2 = high_resolution_clock::now();

		time_spent += duration_cast<duration<double>>(t2 - t1).count();
	}
	// Display execution time
	//cout<<"Time all: "<<time_spent/nbTest<<" s"<<endl;


	std::vector<KeyPoint> res;
	//keep only the 500 best
	std::copy(kps.begin(),kps.begin()+min<int>(kps.size(),500),back_inserter(res));

	// Display the score image
	{
		char buf[100];
		sprintf(buf,"binary_res.png");
		if (ext != NULL)
			sprintf(buf,"binary_res_%s.png",ext);

		double minVal, maxVal;
		minMaxLoc(*score, &minVal, &maxVal);
		double range = maxVal;
		*score = (*score) / range;
		cv::imwrite(buf,*score*255);
	}	

	return res;	
}

vector<KeyPoint> test_fast(const Mat &I,const string &pathFilter, const int &nbTest = 1, Mat* score = NULL)
{
	using namespace std::chrono;
	using namespace cv;
 	high_resolution_clock::time_point t1, t2;
 	std::vector<KeyPoint> kps;
 	double time_spent = 0;



	// Run multiple times to measure average runtime
	for (int i =0;i<nbTest;i++)
	{
		t1 = high_resolution_clock::now();
		// Run TILDE
	    kps = getTILDEKeyPoints_fast(I, pathFilter,true,-std::numeric_limits<float>::infinity(),score);
		t2 = high_resolution_clock::now();

		time_spent += duration_cast<duration<double>>(t2 - t1).count();
	}
	// Display execution time
	//cout<<"Time all: "<<time_spent/nbTest<<" s"<<endl;


	std::vector<KeyPoint> res;
	//keep only the 100 best
	std::copy(kps.begin(),kps.begin()+min<int>(kps.size(),500),back_inserter(res));	

	return res;	
}

void dumpToFile(std::vector<cv::KeyPoint> res, std::string res_fn){
  std::ofstream fs(res_fn.c_str());
  fs << res.size() << " " << 0 << "\n";
  for (int i=0;i<res.size();i++){
    fs << res[i].pt.x << " " << res[i].pt.y << "\n";
    //fs << res[i].pt << "\n";
  }
  fs.close();
}

#define FILTER_DIR "methods/tilde/c++/Lib/filters/"
#define DATA_DIR "datasets/hpatches-sequences-release/"
#define W 640
#define H 480
#define MAX_IMG_NUM 6
#define DEBUG 0
#define SCENE_NUM 116

int main(int argc,char** argv)
{
	using namespace std::chrono;
	using namespace cv;
	string pathFilter;

  if(argc==1){
    std::cout << "Usage:" << std::endl;
    std::cout << "1: trials (int)" << std::endl;
    exit(1);
  }

  if(argc!=2){
    std::cout << "Error: Bad number of arguments" << std::endl;
    exit(1);
  }
  std::string res_dir = "res/tilde/" + std::string(argv[1]) + "/";
  std::cout << res_dir << std::endl;

  // load scene list
  std::string scene_list_fn = "meta/list/img_hp.txt";
  //std::cout << scene_list_fn << std::endl;
  std::vector<std::string> scene_list;
  std::string scene_name="";
  std::ifstream fs;
  fs.open(scene_list_fn.c_str());
  if (fs.is_open()){
    for (int i=0;i<SCENE_NUM;i++){
      fs >> scene_name;
      scene_list.push_back(scene_name);
      //std::cout << scene_name << std::endl;
    }
  }
  else{
    std::cout << "Error: failed to open file " << scene_list_fn << std::endl;
    exit(1);
  }
  fs.close();
  
  std::string res_fn = "";
	try
  {
    // process scenes
    for (int i=0; i<SCENE_NUM; i++){
      scene_name = scene_list[i];
      std::cout << scene_name << std::endl;
      std::string input_directory = std::string(DATA_DIR) + scene_name + "/";

      for (int img_id=1; img_id<(MAX_IMG_NUM+1);img_id++){
        std::stringstream ss;
        ss << img_id;
        std::string img_root_fn = ss.str();
        std::string img_fn = input_directory + img_root_fn + ".ppm";
        //std::cout << "img_fn: " << img_fn << std::endl;

        // Load test image
        Mat I = imread(img_fn);
        if (I.data == 0){
          std::cout << "Error: could not find " << img_fn << std::endl;
          continue;
        }

        cv::resize(I, I, cv::Size(W,H), 0,0, CV_INTER_AREA);
        //std::cout << I.cols << " " << I.rows << std::endl;

        //cout<<"Process image without approximation (Chamonix filter):"<<endl;
        // Path to the TILDE filter
        // Initialize the score image
        pathFilter = std::string(FILTER_DIR) + "Chamonix.txt";
        Mat score1 = Mat::zeros(I.rows,I.cols,CV_32F);
        vector<KeyPoint> kps1 = testAndDump(I,pathFilter,1,NULL, &score1);
        Mat ImgKps1;
        drawKeypoints(I, kps1, ImgKps1);
        if(DEBUG){          
          cv::imshow("keypoints without approximation",ImgKps1);
          cv::imshow("score without approximation",score1);
        }
        // save to file
        res_fn = std::string(res_dir) + scene_name + "/wo/" + img_root_fn + ".txt";
        std::cout << res_fn << std::endl;
        dumpToFile(kps1, res_fn);
        

        ////cout<<"Process Image with approximation (Chamonix filter):"<<endl;
        //// Path to the TILDE approx filter
        //pathFilter = std::string(FILTER_DIR) + "Chamonix24.txt";
        //Mat score2 = Mat::zeros(I.rows,I.cols,CV_32F);
        //vector<KeyPoint> kps2 = testAndDump(I,pathFilter,1,"n_approx", &score2);
        //Mat ImgKps2;
        //drawKeypoints(I, kps2, ImgKps2);
        //if(DEBUG){          
        //  cv::imshow("keypoints with approximation",ImgKps2);
        //  cv::imshow("image with approximation",normalizeScore(score2));
        //}
        //res_fn = std::string(res_dir) + scene_name + "/w24/" + img_root_fn + ".txt";
        //dumpToFile(kps2, res_fn);


        ////cout<<"Process Image with approximation (Chamonix filter) fast:"<<endl;
        //// Path to the TILDE approx filter
        //pathFilter = std::string(FILTER_DIR) + "Chamonix24.txt";
        //Mat score3 = Mat::zeros(I.rows,I.cols,CV_32F);
        //vector<KeyPoint> kps3 = test_fast(I,pathFilter,1, &score3);
        //Mat ImgKps3;
        //drawKeypoints(I, kps3, ImgKps3);
        //res_fn = std::string(res_dir) + scene_name + "/w24_fast/" + img_root_fn + ".txt";
        //dumpToFile(kps3, res_fn);
        //if(DEBUG){
        //  cv::imshow("keypoints with approximation fast",ImgKps3);
        //  cv::imshow("image with approximation fast",normalizeScore(score3));
        //  cv::waitKey(0);
        //}
      }
    }
  }
	catch (std::exception &e) {
		cout<<"ERROR: "<<e.what()<<"\n";
	}

	return 0;
}
// 
// test.cpp ends here
