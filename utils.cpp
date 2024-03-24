/*
    Problem Set 3: Utils
    Joseph Nelson Farrell & Harshil Bhojwani  
    Spring 2024
    CS 5330 Northeastern 
    Professor Bruce Maxwell, PhD
    
    This file contians a library of utility functions used in object recognition
*/

#include <cstdio>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <sstream>

#define SSD(a, b) ( ((int)a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) );

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// Grey Scale ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: greyscale
    Parameters: 2 cv::Mat
        src: the soruce image
        dst: the destination image

    This is simple greyscale filter function.
    This is actually NOT exactly the same as project one greyscale
*/ 
int greyscale( cv::Mat &src, cv::Mat &dst ) { // pass srcs by reference

    // allocate space for the dst src
    src.copyTo( dst ); // makes a copy of the original data

    // update the values in the BGR vector
    for(int i = 0; i<src.rows;i++) { // rows
        for(int j = 0; j<src.cols;j++){ // cols
            int new_val =  (0.114 * src.at<cv::Vec3b>(i, j)[0])+  (0.587 * src.at<cv::Vec3b>(i, j)[1]) + (0.299 * src.at<cv::Vec3b>(i, j)[2]);
            dst.at<cv::Vec3b>(i, j)[0] = new_val;
            dst.at<cv::Vec3b>(i, j)[1] = new_val;
            dst.at<cv::Vec3b>(i, j)[2] = new_val; 
            }
        }
        return(0);
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Gaussian Blur ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: gaussian_blur
    Parameters: 2 cv::Mat
        src: the soruce image
        dst: the destination image

    This function will apply a blur filter using the integer approximation of a Gaussian.
    It will employ seperable 1x5 filters, i.e., ([1 2 4 2 1]) vertically and horizontally.
    It will also use the pointer method.
*/ 
int gaussian_blur(cv::Mat &src, cv::Mat &dst) {

    // allocate memory for the dst and create a temp matrix
    src.copyTo(dst);
    cv::Mat inter_med(src.rows, src.cols, CV_8UC3);

    // iterate over rows
    for (int i = 2; i < src.rows - 2; i++) {

        // assign pointers
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dptr = inter_med.ptr<cv::Vec3b>(i);

        // iterate over cols
        for (int j = 2; j < src.cols - 2; j++) {

            // iterate over channels
            for (int k = 0; k < src.channels(); k++) {

                // apply first 1D filter
                int sum = (ptr[j - 2][k]) + (2 * ptr[j - 1][k]) + (4 * ptr[j][k]) + (2 * ptr[j + 1][k]) + (ptr[j + 2][k]);

                // scale scale result
                sum /= 10;

                // assign new value to dst
                dptr[j][k] = sum;
            }
        }
    }
    // iterate over rows
    for (int i = 2; i < dst.cols - 2; i++) {

        // assign pointers
        cv::Vec3b *dptr_up2 = inter_med.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b *dptr_up = inter_med.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *dptr_md = inter_med.ptr<cv::Vec3b>(i);
        cv::Vec3b *dptr_dn = inter_med.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b *dptr_dn2 = inter_med.ptr<cv::Vec3b>(i + 2);

        // iterate over cols
        for (int j = 2; j < dst.rows - 2; j++) {

            // channels
            for (int k = 0; k < dst.channels(); k++) {

                // apply second 1D filter
                int sum = (dptr_up2[j][k]) + (2 * dptr_up[j][k]) + (4 * dptr_md[j][k]) + (2 * dptr_dn[j + 1][k]) + (dptr_dn2[j + 2][k]);

                // scale result
                sum /= 10;

                // assign to dst
                dst.at<cv::Vec3b>(i, j)[k] = static_cast<uchar>(sum);;
            }
        }
    }
    return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// erosionRect //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: erosionRect
    Parameters: 1 cv::Mat
        src: the soruce image
    Returns: None

    This function perfroms 8-conntected erosion
*/ 
int erosionRect( cv::Mat &src) {
   
   cv::Mat dst = cv::Mat::zeros( src.size(), CV_8UC1);
   src.copyTo(dst);

   dst.setTo(cv::Scalar(0));

    for(int i=1;i<src.rows-1;i++){
        uchar* ptr = src.ptr<uchar>(i);
        for(int j=1;j<src.cols-1;j++){
            if(ptr[j]==0){
            
               dst.at<uchar>(i,j+1)=0;
               dst.at<uchar>(i,j-1)=0;
               dst.at<uchar>(i-1,j)=0;
               dst.at<uchar>(i-1,j+1)=0;
               dst.at<uchar>(i-1,j-1)=0;
               dst.at<uchar>(i+1,j)=0;
               dst.at<uchar>(i+1,j+1)=0;
               dst.at<uchar>(i+1,j-1)=0;
                
            }

        }
    }
    dst.copyTo(src);
    return (0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// erosionCross //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: erosionRect
    Parameters: 1 cv::Mat
        src: the soruce image
    Returns: None

    This function performs 4-conntected erosion
*/ 
int erosionCross( cv::Mat &src) {
   cv::Mat dst = cv::Mat::zeros( src.size(), CV_8UC1);
   src.copyTo(dst);

   dst.setTo(cv::Scalar(255));

    for(int i=1;i<src.rows-1;i++){
        uchar* ptr = src.ptr<uchar>(i);
        for(int j=1;j<src.cols-1;j++){
            if(ptr[j]==0){
                
                    dst.at<uchar>(i,j+1)=0;
                    dst.at<uchar>(i,j-1)=0;
                    dst.at<uchar>(i-1,j)=0;
                    dst.at<uchar>(i+1,j)=0;
                }
            }

        }
        dst.copyTo(src);
        return (0);

    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// dilateCross //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: dilateCross
    Parameters: 1 cv::Mat
        src: the soruce image
    Returns: None

    This function perfroms 4-conntected dilation
*/ 
int dilateCross( cv::Mat &src) {
   cv::Mat dst = cv::Mat::zeros( src.size(), CV_8UC1);
   src.copyTo(dst);

   dst.setTo(cv::Scalar(0));

    for(int i=1;i<src.rows-1;i++){
        uchar* ptr = src.ptr<uchar>(i);
        for(int j=1;j<src.cols-1;j++){
            if(ptr[j]==255 ){
                    dst.at<uchar>(i,j+1)=255;
                    dst.at<uchar>(i,j-1)=255;
                    dst.at<uchar>(i-1,j)=255;
                    dst.at<uchar>(i+1,j)=255;
            }
                
        }
    }
    dst.copyTo(src);
    return (0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// dilateRect //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: dilateCross
    Parameters: 1 cv::Mat
        src: the soruce image
    Returns: None

    This function perfroms 8-conntected dilation
*/ 
int dilateRect( cv::Mat &src) {

   cv::Mat dst = cv::Mat::zeros( src.size(), CV_8UC1);
   src.copyTo(dst);

   dst.setTo(cv::Scalar(0));

    for(int i=1;i<src.rows-1;i++){
        uchar* ptr = src.ptr<uchar>(i);
        for(int j=1;j<src.cols-1;j++){
            if(ptr[j]==255){
            
               dst.at<uchar>(i,j+1)=255;
               dst.at<uchar>(i,j-1)=255;
               dst.at<uchar>(i-1,j)=255;
               dst.at<uchar>(i-1,j+1)=255;
               dst.at<uchar>(i-1,j-1)=255;
               dst.at<uchar>(i+1,j)=255;
               dst.at<uchar>(i+1,j+1)=255;
               dst.at<uchar>(i+1,j+1)=255;
                
            }

        }
    }
    dst.copyTo(src);

     return (0);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// KMeans ///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// This code was provided by: Bruce Maxwell

/*
    data: a std::vector of pixels
    means: a std:vector of means, will contain the cluster means when the function returns
    labels: an allocated array of type int, the same size as the data, contains the labels when the function returns
    K: the number of clusters
    maxIterations: maximum number of E-M interactions, default is 10
    stopThresh: if the means change less than the threshold, the E-M loop terminates, default is 0

    Executes K-means clustering on the data

    This is provided by Professor Maxwell
 */
int kmeans( std::vector<cv::Vec3b> &data, std::vector<cv::Vec3b> &means, int *labels, int K, int maxIterations = 100, int stopThresh = 0) {

  // error checking
  if( K > data.size() ) {
    printf("error: K must be less than the number of data points\n");
    return(-1);
  }

  // clear the means vector
  means.clear();

  // initialize the K mean values
  // use comb sampling to select K values
  int delta = data.size() / K;
  int istep = rand() % (data.size() % K);
  for(int i=0;i<K;i++) {
    int index = (istep + i*delta) % data.size();
    means.push_back( data[index] );
  }
  // have K initial means

  // loop the E-M steps
  for(int i=0;i<maxIterations;i++) {

    // classify each data point using SSD
    for(int j=0;j<data.size();j++) {
      int minssd = SSD( means[0], data[j] );
      int minidx = 0;
      for(int k=1;k<K;k++) {
	int tssd = SSD( means[k], data[j] );
	if( tssd < minssd ) {
	  minssd = tssd;
	  minidx = k;
	}
      }
      labels[j] = minidx;
    }

    // calculate the new means
    std::vector<cv::Vec4i> tmeans(means.size(), cv::Vec4i(0, 0, 0, 0) ); // initialize with zeros
    for(int j=0;j<data.size();j++) {
      tmeans[ labels[j] ][0] += data[j][0];
      tmeans[ labels[j] ][1] += data[j][1];
      tmeans[ labels[j] ][2] += data[j][2];
      tmeans[ labels[j] ][3] ++; // counter
    }
    
    int sum = 0;
    for(int k=0;k<tmeans.size();k++) {
      tmeans[k][0] /= tmeans[k][3];
      tmeans[k][1] /= tmeans[k][3];
      tmeans[k][2] /= tmeans[k][3];

      // compute the SSD between the new and old means
      sum += SSD( tmeans[k], means[k] );

      means[k][0] = tmeans[k][0]; // update the mean
      means[k][1] = tmeans[k][1]; // update the mean
      means[k][2] = tmeans[k][2]; // update the mean
    }

    // check if we can stop early
    if( sum <= stopThresh ) {
      break;
    }
  }

  // the labels and updated means are the final values

  return(0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////  Threshold ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: threshold
    Parameters: 2 cv::Mat
        src: the soruce image
        dst: the destination image
    Returns: none

    Functions steps:
    1. Apply a blur filter using the integer approximation of a Gaussian.
    2. Convert to greyscale
    3. Apply kmeans, with k = 2 to dynamically compute the threshold
    4. Convert image to binary
*/ 

int threshold( cv::Mat &src, cv::Mat &dst ) {

    // declare variables
    int ncolors = 2;

    // create dst image
    dst = cv::Mat::zeros( src.size(), CV_8UC1);

    // blur image
    gaussian_blur( src, src);

    // greyscale src
    greyscale( src, src);

    // sample colors from the image using jitter sampling
    // sample one color from each B x B block of the image
    int B = 16;
    std::vector<cv::Vec3b> data;
    for(int i = 16;i < src.rows - B;i += B) {
        for(int j=0;j < src.cols - B;j += B) {
        int jx = rand() % B;
        int jy = rand() % B;
        data.push_back( src.at<cv::Vec3b>(i+jy, j+jx) );
        }
    }

    // instantiate vector to store means
    std::vector<cv::Vec3b> means;

    // labels pointer for kmeans
    int *labels = new int[data.size()];

    if(kmeans( data, means, labels, ncolors ) ) {
      printf("Erro using kmeans\n");
      return(-1);
    }

    // compute the the thresold value
    float mean_1 = means[0][0];
    float mean_2 = means[1][0];
    float threshold = (mean_1 + mean_2) / 2;

    // convert image to binary, i.e., black and white
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            float value = static_cast<float>(src.at<cv::Vec3b>(i, j)[0]);
            if(value > threshold) {
                dst.at<uchar>(i, j) = 0;
            } else {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    
    return(0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Scaled Euclidean /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: scaled_euclidean
    Parameters: 3 std::vector
        obl_feature_vector: the feature vector of the current object
        db_feature_vector: feature vector of object in the db
        std_vector: a vector of std of each feature

    This function will compute the scaled the euclidean distance between two feature vectors.
*/ 
float scaled_euclidean( const std::vector<float> &vector_1, const std::vector<float> &vector_2, const std::vector<float> &vector_3){

    // check that the vector lengths match
    if( vector_1.size() != vector_2.size()){
        printf("Error! Vector lengths must be equal to compute distance!");
        }
    float sum = 0.0;
    for(int i = 0; i < vector_1.size(); i ++) {
            sum += std::pow((vector_1[i] - vector_2[i])/vector_3[i], 2);
        }
    return(sum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////  Cosine Distance    /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: cosine_distance
    Param: 2 std::vector<float>
        vector_1: target feature vector
        vector_2: iamge feature vector
    Returns: 1 float
        cosine_distance: the distance between the two vectors
    
   This function computes the cosine distance metric, i.e., the cosine(theta).
*/
float cosine_distance(const std::vector<float> &vector_1, const std::vector<float> &vector_2) {

    // compute the dot product
    float dot_product = 0.0;

    for (size_t i = 0; i < vector_1.size(); ++i) {
        dot_product += vector_1[i] * vector_2[i];
    }

    // compute the magnitudes
    float length_vector_1 = static_cast<float>(cv::norm(vector_1, cv::NORM_L2, cv::noArray()));
    float length_vector_2 = static_cast<float>(cv::norm(vector_2, cv::NORM_L2, cv::noArray()));

    // compute the cosine similarity
    float cosine_similarity = dot_product / (length_vector_1 * length_vector_2);

    // convert similarity to distance
    float cosine_distance = 1.0f - cosine_similarity;

    return cosine_distance;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////  Create Feature Matrix    ///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: create_feature_matrix
    Param: 1 std::vector<float>, 1 std::string
        labels: a vector to store the object labels from the database
        csv_files_path: path to the csv file with the database 
    Returns: 1 cv::Mat
        feature_matrix: a matrix holding all the values from the database, each row is observation
    
   This function will read in a csv and create a matrix with contents
*/
cv::Mat create_feature_matrix(std::vector<std::string> &labels, std::string & csv_file_path ){

    // declare variables
    std::ifstream file(csv_file_path);
    std::string line;
    std::vector<std::vector<float>> data; 

    // iterate over data in file
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string cell;
        std::vector<float> row_data;
        bool first_col = true;
        
        while (std::getline(line_stream, cell, ',')) {
            if (first_col) {

                // update labels vector for comparisons
                labels.push_back(cell);
                first_col = false;
            } else {
                row_data.push_back(std::stof(cell));
                }
            }
        data.push_back(row_data);
    }

    // create matrix
    cv::Mat feature_matrix(data.size(), data[0].size(), CV_32F); 
    for (int i = 0; i < feature_matrix.rows; ++i) {
        for (int j = 0; j < feature_matrix.cols; ++j) {
            feature_matrix.at<float>(i, j) = data[i][j];
        }
    }

    return(feature_matrix);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////  Compute STD Vector   ///////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: compute_std_vector
    Param: 1 cv::Mat
        feature_matrix: the matrix storing the values from the database
    Returns: 1 std::vector
        std_vector: a vector of feature standard deviations
    
   This function will compute the standard deviations of the cols of a cv::Mat and return a vector of stds
*/
std::vector<float> compute_std_vector( cv::Mat &feature_matrix ) {

    // declare variables
    std::vector<float> std_vector;
    std_vector.reserve(feature_matrix.cols);
    cv::Scalar mean, std;

    // compute std, update vector
    for (int col = 0; col < feature_matrix.cols; ++col) {
        cv::Mat column = feature_matrix.col(col);
        cv::meanStdDev(column, mean, std);
        std_vector.push_back(std[0]);
        }

    return(std_vector);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// getEmbedding   /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Thus function was provided by: Bruce Maxwell

/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug  = 0 ){
  const int ORNet_size = 128;
  cv::Mat padImg;
  cv::Mat blob;
	
  cv::Mat roiImg = src( bbox );
  int top = bbox.height > 128 ? 10 : (128 - bbox.height)/2 + 10;
  int left = bbox.width > 128 ? 10 : (128 - bbox.width)/2 + 10;
  int bottom = top;
  int right = left;
	
  cv::copyMakeBorder( roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0  );
  cv::resize( padImg, padImg, cv::Size( 128, 128 ) );

  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) / 0.5, // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  128,   // subtract mean prior to scaling
			  false, // input is a single channel image
			  true,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!/fc1/Gemm" );

  if(debug) {
    cv::imshow( "pad image", padImg );
    std::cout << embedding << std::endl;
    cv::waitKey(0);
  }

  return(0);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////  get_dnn_embedding   ////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// This code was adapted from code provided by: Bruce Maxwell

/*
    Function: get_dnn_embedding
    Param: 1 cv::Mat, 1 std::vector
        threshold_for_dnn: a threshold image
    Returns: 1 std::vector
        dnn_embedding_vector: dnn a dnn feature vector
    
   This function will take a threshold image and return a dnn feature vector
*/

std::vector<float> get_dnn_embedding_vector( cv::Mat &threshold_for_dnn, std::vector<float> &dnn_embedding_vector){
        
        std::string file_path = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_files/dnn_feature_vectors.csv";
        char file_path_char[256];  
        strcpy(file_path_char, file_path.c_str());

        std::string mod_filename = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/dnnexample/or2d-normmodel-007.onnx";
        cv::dnn::Net net = cv::dnn::readNet( mod_filename );

        cv::Rect bbox( 0, 0, threshold_for_dnn.cols, threshold_for_dnn.rows );

        cv::Mat embedding;
        getEmbedding( threshold_for_dnn, embedding, bbox, net, 0 );

            for (int i = 0; i < embedding.rows; i++) {
                for (int j = 0; j < embedding.cols; j++) {
                    dnn_embedding_vector.push_back(embedding.at<float>(i, j));
                }
            }
        return( dnn_embedding_vector );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////  twoPassAlgo   ///////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    Function: twoPassAlgo
    Param: 2 cv::Mat, 3 std::vectors, 1 int
        src: the source iamge
        dimg: region map image
        size: vector of region sizes
        boundingBoxes: bounding box coordinates
        centriod: center coordinates
    Returns: None
    
   This function will find the connected componenets and compute statistics on the regions
*/

int twoPassAlgo(cv::Mat src, cv::Mat &dimg, int &regionMap, std::vector<int>&size,  std::vector<std::vector<std::pair<int, int>>>&boundingBoxes, std::vector<std::pair<int, int>>&centroid){
    int parent[800];
    parent[0]=0;
     
    dimg = cv::Mat::zeros( src.size(), CV_8UC1 );

    for(int i=1;i<src.rows;i++) {
       
        for(int j=1;j<src.cols;j++) {
        
            if(src.at<uchar>(i, j) > 0 ) { // foreground pixel
              if(dimg.at<uchar>(i-1, j)==0 && dimg.at<uchar>(i, j-1)==0){
                regionMap++;
                dimg.at<uchar>(i, j)=regionMap;
                parent[regionMap]=regionMap;
                
              }else if(dimg.at<uchar>(i-1, j)>0 && dimg.at<uchar>(i, j-1)==0){
                dimg.at<uchar>(i, j)=dimg.at<uchar>(i-1, j);
              }else if(dimg.at<uchar>(i, j-1)>0 && dimg.at<uchar>(i-1, j)==0){
                dimg.at<uchar>(i, j)=dimg.at<uchar>(i, j-1);
              }else if(dimg.at<uchar>(i, j-1)>0 && dimg.at<uchar>(i-1, j)>0){
                if(dimg.at<uchar>(i, j-1)==dimg.at<uchar>(i-1, j)){
                    dimg.at<uchar>(i, j)=dimg.at<uchar>(i, j-1);
                }else if(dimg.at<uchar>(i, j-1)<=dimg.at<uchar>(i-1, j)){
                    dimg.at<uchar>(i, j)=dimg.at<uchar>(i, j-1);
                    dimg.at<uchar>(i-1, j)=dimg.at<uchar>(i, j-1);
                    parent[dimg.at<uchar>(i-1, j)]=dimg.at<uchar>(i-1, j);
                }else{
                    dimg.at<uchar>(i, j)=dimg.at<uchar>(i-1, j);
                    dimg.at<uchar>(i, j-1)=dimg.at<uchar>(i-1, j);
                    parent[dimg.at<uchar>(i, j-1)]=dimg.at<uchar>(i-1, j);
                }
            }
        }
    } 
    }

   for(int i=0;i<dimg.rows;i++) {
        for(int j=0;j<dimg.cols;j++) { 
            if((dimg.at<uchar>(i, j) > 0) && (dimg.at<uchar>(i, j) != parent[dimg.at<uchar>(i, j) ] )){
                dimg.at<uchar>(i, j) = parent[dimg.at<uchar>(i, j)];

            }
            //printf("region = %d\n",dimg.at<uchar>(i, j));
        }
   }
    for(int i=0;i<regionMap;i++){
     int zero = 0;
     size.push_back(zero); 
     //printf("size of region %d =%d\n", i, size[i]);
    }

for(int i=0;i<regionMap;i++){
    std::vector<std::pair<int, int>> point = {{INT_MAX, INT_MAX},{INT_MIN, INT_MIN}};
    boundingBoxes.push_back(point); 
   }

    //printf("statistics calucalted\n");

   
   for(int i=0;i<dimg.rows;i++) {
        for(int j=0;j<dimg.cols;j++){
            int ref=dimg.at<uchar>(i, j);
            if(dimg.at<uchar>(i, j) > 0){  
                size[ref]++;
                int minX = boundingBoxes[ref][0].first;
                int maxX = boundingBoxes[ref][1].first;
                int minY = boundingBoxes[ref][0].second;
                int maxY =  boundingBoxes[ref][1].second;
                if(i<minX){
                minX=i;
                }
               if(i>maxX){
                maxX=i;
               }
               if(j>maxY){
                maxY=j;
               }
               if(j<minY){
                minY=j;
               }
               //printf("Value about to be edited"); 
               boundingBoxes[ref]={ {minX, minY}, {maxX, maxY}}; 
               //printf("Value edited");      
            }
        }
    }
    for(int i=0;i<regionMap;i++){
        int zero1=0;
       std::pair<int, int> zeroPair = std::make_pair(zero1,zero1);
       centroid.push_back(zeroPair);
   }
    for(int i=0;i<regionMap;i++){
    int centerX =  (boundingBoxes[i][0].first + boundingBoxes[i][1].first)/2;
    int centerY = (boundingBoxes[i][0].second + boundingBoxes[i][1].second)/2;
    std::pair<int, int> centerPair = std::make_pair(centerX,centerY);
    centroid[i] = centerPair;
   }
return (1);

}  














    





    













