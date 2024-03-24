/*
    Problem Set 3: Utils
    Joseph Nelson Farrell & Harshil Bhojwani    
    Spring 2024
    CS 5330 Northeastern 
    Professor Bruce Maxwell, PhD
    
    This is the header file for library of helper functions for object_rec.cpp
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>

#ifndef UTILS_H
#define UTILS_H

#define SSD(a, b) ( ((int)a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) )

int kmeans( std::vector<cv::Vec3b> &data, std::vector<cv::Vec3b> &means, int *labels, int K, int maxIterations = 100, int stopThresh = 0); 

int gaussian_blur(cv::Mat &src, cv::Mat &dst);

int greyscale( cv::Mat &src, cv::Mat &dst );

int threshold( cv::Mat &src, cv::Mat &dst );

float scaled_euclidean( const std::vector<float> &vector_1, const std::vector<float> &vector_2, const std::vector<float> &vector_3);

cv::Mat create_feature_matrix(std::vector<std::string> &labels, std::string & csv_file_path );

std::vector<float> compute_std_vector( cv::Mat &feature_matrix );

int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug  = 0 );

float cosine_distance(const std::vector<float> &vector_1, const std::vector<float> &vector_2);

std::vector<float> get_dnn_embedding_vector( cv::Mat &threshold_for_dnn, std::vector<float> &dnn_embedding_vector);

int dilateRect( cv::Mat &src);

int dilateCross( cv::Mat &src);

int erosionCross( cv::Mat &src);

int erosionRect( cv::Mat &src);

int twoPassAlgo(cv::Mat src, cv::Mat &dimg, int &regionMap, std::vector<int>&size,  std::vector<std::vector<std::pair<int, int>>>&boundingBoxes, std::vector<std::pair<int, int>>&centroid);

float SSD2(int MinX, int MinY, int MaxX, int MaxY);

#endif