/*
    Problem Set 3: Object Rec
    Joseph Nelson Farrell & Harshil Bhojwani  
    Spring 2024
    CS 5330 Northeastern 
    Professor Bruce Maxwell, PhD
    
    This file contains a program that will process images, generate feature vectors, and create a data base of objects in the
    images with their label and a feature vector. It can create two different feature vectors, one containing classic features 
    and another containing a DNN embedding. It can also classify objects that are in the database.

*/
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {

    cv::Mat src;
    char filename[256];
    int ncolors = 2;
    char lastKey = 0;
    bool save_flag = false; 

    // error checking
    if(argc < 1) {
        printf("usage: %s <image filename>",  argv[0]);
        return(-1);
    }

    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(1);
    if ( !capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get image properties
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // print results
    printf("Expected size: %d %d\n", refS.width, refS.height);

    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// Setup Confusion Matrices ////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // classic features
    std::string csv_file_path = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_files/feature_vectors.csv";
    std::vector<std::string> con_mat_labels;
    cv::Mat feature_matrix = create_feature_matrix( con_mat_labels, csv_file_path );           

    std::vector<std::string> confusion_labels = con_mat_labels;
    std::sort(confusion_labels.begin(), confusion_labels.end());
    auto it = std::unique(confusion_labels.begin(), confusion_labels.end());
    confusion_labels.erase(it, confusion_labels.end());

    // map label to index
    std::map<std::string, int> label_to_index;
    for (int i = 0; i < confusion_labels.size(); ++i) {
        label_to_index[confusion_labels[i]] = i;
    }
    // instantiate confustion matrix
    int num_classes = confusion_labels.size();
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // dnn
    std::string dnn_csv_file_path = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_files/dnn_feature_vectors.csv";
    std::vector<std::string> dnn_con_mat_labels;
    cv::Mat dnn_feature_matrix = create_feature_matrix( dnn_con_mat_labels, dnn_csv_file_path );           

    std::vector<std::string> dnn_confusion_labels = dnn_con_mat_labels;
    std::sort(dnn_confusion_labels.begin(), dnn_confusion_labels.end());
    auto dnn_it = std::unique(dnn_confusion_labels.begin(), dnn_confusion_labels.end());
    dnn_confusion_labels.erase(dnn_it, dnn_confusion_labels.end());

    // map label to index
    std::map<std::string, int> dnn_label_to_index;
    for (int i = 0; i < dnn_confusion_labels.size(); ++i) {
        dnn_label_to_index[dnn_confusion_labels[i]] = i;
    }
    // instantiate confustion matrix
    int dnn_num_classes = dnn_confusion_labels.size();
    std::vector<std::vector<int>> dnn_confusion_matrix(dnn_num_classes, std::vector<int>(dnn_num_classes, 0));
    
    for(;;) {

        // this will get a new src from the camera, treat as a stream
        *capdev >> src;
        if( src.empty() ) {
            printf( "src is empty!!\n");
            break;
        }
      
        // check for keystrokes
        char key = cv::waitKey(1) & 0xFF;

        // save the last keystroke
        if (key != -1) {
            if (key == 's'){
                save_flag = true;
            } else { lastKey = key;}
        }

        // save the last keystroke
        if( lastKey == 'q') {
            break;
        } 

        cv::Mat color_region_display(src.rows, src.cols, src.type());
        cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

        //cv::imshow("Source Display", src);
    
        // read the file
        if( src.data == NULL ) {
            printf("error: unable to read filename %s\n", filename);
            return(-2);
        }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Thresholding /////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // threshold the image
        threshold( src, dst );
        //cv::imshow("Threshold Image", dst);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Erosion and Dilation /////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        for(int i = 0; i < 1; i++){
            dilateRect( dst );
        }
        cv::imshow("Dilate", dst);
        for(int i = 0; i < 2; i++){
            erosionCross( dst );
        }
        //cv::imshow("Post Erode ", dst);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Connected Compontents ////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // these variables will be used to disregard regions
        int min_area = 2000; 

        // variables to store region information
        std::vector<int> kept_labels;
        cv::Mat labels, stats, centroids;

        // find connected components
    int num_labels = cv::connectedComponentsWithStats(dst, labels, stats, centroids);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////// filter out components that are too small ////////////////////////////////////////////////////////////

        for (int label = 0; label < num_labels; label++) {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            int x = stats.at<int>(label, cv::CC_STAT_LEFT);
            int y = stats.at<int>(label, cv::CC_STAT_TOP);
            int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
            cv::Point centroid = cv::Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1));        
            bool touches_border = x == 0 || y == 0 || (x + width) == dst.cols || (y + height) == dst.rows;

            if ((area >= min_area) && !touches_border) {
                kept_labels.push_back(label); 
            }
        }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////// find center most components /////////////////////////////////////////////////////////////////////////////

        // find the center of the image
        cv::Point image_center(dst.cols / 2, dst.rows / 2 );

        // pair vector to store distances and labels
        std::vector<std::pair<int, double>> distances;

        // number of components
        int components = centroids.rows;

        // compute distance to center
        for(int label : kept_labels){
            double x = centroids.at<double>(label, 0);
            double y = centroids.at<double>(label, 1);
            cv::Point compontent_center(x, y);
            double dist = cv::norm(compontent_center - image_center);
            distances.push_back(std::make_pair(dist, label));
        }

        // sort the distances array & extract the most center label
        int center_most_label = 0; // used prevent seg fault if no labels are kept
        std:sort(distances.begin(), distances.end());
        if (!distances.empty()) {
            center_most_label = distances.front().second; 
        } else {
            continue;
        }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////// color the kept region ////////////////////////////////////////////////////////////////////////////////////

        // declare mat for store color image
        cv::Mat components_colored = cv::Mat::zeros(dst.size(), CV_8UC3);
        cv::Mat threshold_for_dnn = cv::Mat::zeros(dst.size(), CV_8UC1); 

        // declare pair vector to store color label pairs for kept labels
        std::vector<std::pair< int, cv::Vec3b >> colors;

        // generate colors for kept labels
 
        for(int i = 0; i < kept_labels.size(); i++) {
            int r = 255 % (i + 1);
            int g = 125; 
            int b = 125 % i; 
            colors.push_back(std::make_pair(kept_labels[i], cv::Vec3b(b, g, r)));
        }

    // this will create a color image where each of the kept regoins has a different color
    // rows
    for(int i = 0; i < labels.rows; i++) {

        // cols
        for(int j = 0; j < labels.cols; j++) {

            // get label from location
            int label = labels.at<int>(i, j);

            // find color associated with label, color image
            auto it = std::find_if(colors.begin(), colors.end(), 
                [label](const std::pair<int, cv::Vec3b>& element) { return element.first == label; });
            if(it != colors.end()) {
                components_colored.at<cv::Vec3b>(i, j) = it->second;
                threshold_for_dnn.at<uchar>(i, j) = 255;
            }
        }
    }
    //cv::imshow("Segmented Image", components_colored);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Central Moments //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cv::Moments m = cv::moments(labels == center_most_label);

    double center_x = m.m10 / m.m00;
    double center_y = m.m01 / m.m00;

    double m_20 = m.mu20 - (m.m10 * m.m10 / m.m00);
    double m_02 = m.mu02 - (m.m01 * m.m01 / m.m00);
    double m_11 = m.mu11 - (m.m10 * m.m01 / m.m00);

    double theta = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02);

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double m_sin_theta = -std::sin(theta);

    cv::Point2f e1(cos_theta, sin_theta);
    cv::Point2f e2(-sin_theta, cos_theta);

    cv::Point2f mean_pix(center_x, center_y);

    double length = 100; 

    cv::Point2f end_pix_e1(center_x + length * cos_theta, center_y + length * sin_theta);
    cv::Point2f end_pix_e2(center_x + length * -sin_theta, center_y + length * cos_theta);

    cv::Mat features_display_stream = src.clone();

    // draw arrows
    cv::arrowedLine(components_colored, mean_pix, end_pix_e1, cv::Scalar(255, 255, 255), 4);
    cv::arrowedLine(components_colored, mean_pix, end_pix_e2, cv::Scalar(255, 255, 255), 4);

    float max_e1 = std::numeric_limits<float>::lowest();
    float min_e1 = std::numeric_limits<float>::max();
    float max_e2 = std::numeric_limits<float>::lowest();
    float min_e2 = std::numeric_limits<float>::max();

    int num_pixels = 0; // this will be used later to compute the percent filled of the bounding box
    
     for(int y = 0; y < labels.rows; y++){
        for(int x = 0; x < labels.cols; x++){
            int label = labels.at<int>(y, x);
            if(labels.at<int>(y, x) == center_most_label) {

                num_pixels++;

                float proj_e1 = ((x - center_x) * cos_theta) + ((y - center_y) * sin_theta);
                float proj_e2 = ((x - center_x) * (-sin_theta)) + ((y - center_y) * cos_theta);

                max_e1 = std::max(max_e1, proj_e1);
                min_e1 = std::min(min_e1, proj_e1);
                max_e2 = std::max(max_e2, proj_e2);
                min_e2 = std::min(min_e2, proj_e2);
            }
        }
    }

    // calc corners
    cv::Point2f top_right(mean_pix + min_e1 * e1 + min_e2 * e2); // top right
    cv::Point2f bottom_right(mean_pix + max_e1 * e1 + min_e2 * e2); // bottom right
    cv::Point2f top_left(mean_pix + min_e1 * e1 + max_e2 * e2); // top left
    cv::Point2f bottom_left(mean_pix + max_e1 * e1 + max_e2 * e2); // bottom left

    // draw rect
    cv::line(components_colored, bottom_left, top_left, cv::Scalar(0, 255, 255), 2);
    cv::line(components_colored, top_left, top_right, cv::Scalar(0, 255, 255), 2);
    cv::line(components_colored, bottom_right, bottom_left, cv::Scalar(0, 255, 255), 2);
    cv::line(components_colored, top_right, bottom_right, cv::Scalar(0, 255, 255), 2);

    // cirlcle corner points
    cv::circle(components_colored, bottom_left, 10,cv::Scalar(255, 255, 255), 2);
    cv::circle(components_colored, top_right, 10,cv::Scalar(0, 255, 0), 2);
    cv::circle(components_colored, top_left, 10,cv::Scalar(0, 0, 255), 2);
    cv::circle(components_colored, bottom_right,10, cv::Scalar(255, 0, 0), 2);

    cv::imshow("Axes and Bounding Box", components_colored);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Calc Features ////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // compute height width ratio
    double height = sqrt(pow(top_left.x - bottom_left.x, 2) + pow(top_left.y - bottom_left.y, 2));
    double width = sqrt(pow(bottom_left.x - bottom_right.x, 2) + pow(bottom_left.y - bottom_right.y, 2));
    float w_h_ratio = static_cast<float>(width) / height;

    // compute percent filled
    float total_area = height * width;
    double percent_filled = num_pixels / total_area;

    // format percent fill for display
    std::ostringstream percent_fill_label;
    percent_fill_label << std::fixed << std::setprecision(2) << percent_filled; 
    std::string percent_fill_text = "Percent Filled: " + percent_fill_label.str();

    // format height width ratio for display
    std::ostringstream hwr_label;
    hwr_label << std::fixed << std::setprecision(2) << w_h_ratio; 
    std::string hwr_text = "Height Width Ratio: " + hwr_label.str();

    // text postioning
    int percent_fill_x = 50;
    int percent_fill_y = 275;
    int hwr_x = 50;
    int hwr_y = 325;

    // add ratio to image
    cv::putText(components_colored, percent_fill_text, cv::Point(percent_fill_x, percent_fill_y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    cv::putText(components_colored, hwr_text, cv::Point(hwr_x, hwr_y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    cv::imshow("Features Display", components_colored);


    // create feature vector for object
    std::vector<float> feature_vector;
    feature_vector.push_back(percent_filled);
    feature_vector.push_back(w_h_ratio);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Create DB csv ////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if( key == 't') {
            std::cout << "\n******************************************"<< std::endl;
            std::cout << "Entering Training Mode: Classical Feature Vectors"<< std::endl;
            std::cout << "******************************************\n"<< std::endl;
            std::cout << "What is the label of this item? Please enter the name:"<< std::endl;
            std::string label;
            std::getline(std::cin, label); 

            std::string file_path = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_files/feature_vectors.csv";
            char file_path_char[256];  
            strcpy(file_path_char, file_path.c_str());
            append_image_data_csv(file_path_char, const_cast<char*>(label.c_str()), feature_vector, 0);

            // mode
            std::string mode_text = "Training Mode: Classic";
            std::string message_text = "Object Entered into Data Base!!!";

            // distance
            std::ostringstream distance_label;
            std::string distance_text = " Min Distance: " + distance_label.str();

            // label 
            std::string object_name_text = "Object Class: " + label;

            // text postioning
            int obj_x = src.cols / 3;
            int obj_y = src.rows - 50;

            // add ratio to image
            cv::putText(components_colored, object_name_text, cv::Point(obj_x, obj_y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
            cv::putText(components_colored, mode_text, cv::Point(src.cols / 4, 100), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
            cv::putText(components_colored, message_text,cv::Point(src.cols / 4, 175), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Classified", components_colored);
            cv::waitKey(0);
            cv::destroyWindow("Classified");

            std::cout << "Complete!!! Data enter into database."<< std::endl;
        }

        if( key == 'd') {
            std::cout << "\n******************************************"<< std::endl;
            std::cout << "Entering Training Mode: DNN"<< std::endl;
            std::cout << "******************************************\n"<< std::endl;
            std::cout << "What is the label of this item? Please enter the name:"<< std::endl;
            std::string label;
            std::getline(std::cin, label); 

            std::string file_path = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_files/dnn_feature_vectors.csv";
            char file_path_char[256];  
            strcpy(file_path_char, file_path.c_str());


            std::vector<float> dnn_embedding_vector;
            get_dnn_embedding_vector( threshold_for_dnn, dnn_embedding_vector );
            append_image_data_csv(file_path_char, const_cast<char*>(label.c_str()), dnn_embedding_vector, 0);

            // mode
            std::string mode_text = "Training Mode: DNN";
            std::string message_text = "Object Entered into Data Base!!!";

            // distance
            std::ostringstream distance_label;
            std::string distance_text = " Min Distance: " + distance_label.str();

            // label 
            std::string object_name_text = "Object Class: " + label;

            // text postioning
            int obj_x = src.cols / 3;
            int obj_y = src.rows - 50;

            // add ratio to image
            cv::putText(components_colored, object_name_text, cv::Point(obj_x, obj_y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
            cv::putText(components_colored, mode_text, cv::Point(src.cols / 4, 100), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
            cv::putText(components_colored, message_text,cv::Point(src.cols / 4, 175), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Classified", components_colored);
            cv::waitKey(0);
            cv::destroyWindow("Classified");
            std::cout << "Complete!!! Data enter into database."<< std::endl;
        }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////// Classification ///////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if( key == 'c') {            
            std::string csv_file_path = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_files/feature_vectors.csv";
            std::vector<std::string> labels;
            cv::Mat feature_matrix = create_feature_matrix( labels, csv_file_path );
            std::vector<float> std_vector;
            std_vector = compute_std_vector( feature_matrix );

            std::vector<float> distances_vector;
            for(int i = 0; i < feature_matrix.rows; i++){
                std::vector<float> feature_vector_2 = feature_matrix.row(i);
                float dist = scaled_euclidean( feature_vector, feature_vector_2, std_vector );
                distances_vector.push_back( dist );
            }

            // get the index of min element
            auto min_element = std::min_element(distances_vector.begin(), distances_vector.end());
            int min_index = std::distance(distances_vector.begin(), min_element);
            
            if(distances_vector[min_index] > 0.15){
                std::string mode_text = "Classification Mode: Classic Features";
                std::string message_text = "NOT IN DATA BASE...please classify!";
                cv::putText(src, message_text, cv::Point(50 , src.rows - 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
                cv::putText(src, mode_text, cv::Point(src.cols / 5, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 3);
                cv::imshow("Classified", src);
                cv::waitKey(0);
                cv::destroyWindow("Classified");
                continue;
                }
            std::string true_label;
            std::cout << "What is the true label?"<< std::endl;
            std::getline(std::cin, true_label); 
            std::string predicted_label = labels[min_index]; 

            // Update the confusion matrix
            int true_label_index = label_to_index[true_label];
            int predicted_label_index = label_to_index[predicted_label];        
            confusion_matrix[true_label_index][predicted_label_index]++;

            // format text for display

            // mode
            std::string mode_text = "Classification Mode: Classic Features";

            // distance
            std::ostringstream distance_label;
            distance_label << std::fixed << std::setprecision(2) << distances_vector[min_index]; 
            std::string distance_text = " Min Distance: " + distance_label.str();

            // label 
            std::string object_name_text = "Predicted Class: " + labels[min_index];
            std::string true_name_text = "True Class: " + true_label;

            // text postioning
            int distance_label_x = 50;
            int distnace_label_y = 225;
            int obj_x = src.cols / 2;
            int obj_y = src.rows - 50;

            // add ratio to image
            cv::putText(src, distance_text, cv::Point(distance_label_x, distnace_label_y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(src, object_name_text, cv::Point(obj_x, obj_y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
            cv::putText(src, true_name_text, cv::Point(50 , obj_y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
            cv::putText(src, mode_text, cv::Point(src.cols / 5, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 3);
            cv::imshow("Classified", src);
            cv::waitKey(0);
            cv::destroyWindow("Classified");

        }

        if( key == 'p') {            
            std::string csv_file_path = "/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_files/dnn_feature_vectors.csv";
            std::vector<std::string> labels;
            cv::Mat feature_matrix = create_feature_matrix( labels, csv_file_path );

            std::vector<float> dnn_embedding_vector;

            get_dnn_embedding_vector( threshold_for_dnn, dnn_embedding_vector );

            std::vector<float> distances_vector;
            for(int i = 0; i < feature_matrix.rows; i++){
                std::vector<float> feature_vector_2 = feature_matrix.row(i);
                float dist = cosine_distance( dnn_embedding_vector, feature_vector_2 );
                distances_vector.push_back( dist );
            }

            // find shortest dist
            auto minIt = std::min_element(distances_vector.begin(), distances_vector.end());

            // find index of shortest dist
            int min_index = std::distance(distances_vector.begin(), minIt);

            if(distances_vector[min_index] > 0.15){
                std::string mode_text = "Classification Mode: DNN";
                std::string message_text = "NOT IN DATA BASE...please classify!";
                cv::putText(src, message_text, cv::Point(50 , src.rows - 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
                cv::putText(src, mode_text, cv::Point(src.cols / 5, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 3);
                cv::imshow("Classified", src);
                cv::waitKey(0);
                cv::destroyWindow("Classified");
                continue;
                }

            std::string true_label;
            std::cout << "What is the true label?"<< std::endl;
            std::getline(std::cin, true_label); 
            std::string predicted_label = labels[min_index];

            // Update the confusion matrix
            int true_label_index = dnn_label_to_index[true_label];
            int predicted_label_index = dnn_label_to_index[predicted_label];        
            dnn_confusion_matrix[true_label_index][predicted_label_index]++;

            // format text for display
            // mode
            std::string mode_text = "Classification Mode: DNN";

            // distance
            std::ostringstream distance_label;
            distance_label << std::fixed << std::setprecision(2) << distances_vector[min_index]; 
            std::string distance_text = " Min Distance: " + distance_label.str();

            // label 
            std::string object_name_text = "Predicted Class: " + labels[min_index];
            std::string true_name_text = "True Class: " + true_label;

            // text postioning
            int distance_label_x = 50;
            int distnace_label_y = 225;
            int obj_x = components_colored.cols / 2;
            int obj_y = components_colored.rows - 50;

            // add ratio to image
            cv::putText(src, distance_text, cv::Point(distance_label_x, distnace_label_y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(src, object_name_text, cv::Point(obj_x, obj_y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
            cv::putText(src, true_name_text, cv::Point(50 , obj_y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
            cv::putText(src, mode_text, cv::Point(src.cols / 5, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 3);
            cv::imshow("Classified", src);
            cv::waitKey(0);
            cv::destroyWindow("Classified");
        }
    }

    std::string response;
    std::cout << "Do you want to display the confusion matrix?"<< std::endl;
    std::getline(std::cin, response);
    if(response == "yes"){
        std::cout << "This is the order of the rows and cols: \n"<< std::endl;
        for(const auto& str : confusion_labels) {
            std::cout << str << " "; 
                }
        std::cout << "\n" << std::endl;
        for(const auto& row : confusion_matrix) {
            for(int val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    std::string response_2;
    std::cout << "Do you want to display the dnn confusion matrix?"<< std::endl;
    std::getline(std::cin, response_2);
    if(response_2 == "yes"){
        std::cout << "This is the order of the rows and cols: \n"<< std::endl;
        for(const auto& str : dnn_confusion_labels) {
            std::cout << str << " "; 
                }
        std::cout << "\n" << std::endl;
        for(const auto& row : dnn_confusion_matrix) {
            for(int val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
        
    delete capdev;
    return(0);
}