/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef DATA_PREPROCESS_HPP
#define DATA_PREPROCESS_HPP

#include "CustomMsg.h"
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

using namespace std;
using namespace cv;

class DataPreprocess
{
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input_;

    cv::Mat img_input_;

    DataPreprocess(Params &params)
        : cloud_input_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        string bag_path = params.bag_path;
        string image_path = params.image_path;
        string lidar_topic = params.lidar_topic;

        img_input_ = cv::imread(params.image_path, cv::IMREAD_UNCHANGED);
        if (img_input_.empty()) 
        {
            std::string msg = "Loading the image " + image_path + " failed";
            ROS_ERROR_STREAM(msg.c_str());
            return;
        }

        std::fstream file_;
        file_.open(bag_path, ios::in);
        if (!file_) 
        {
            std::string msg = "Loading the rosbag " + bag_path + " failed";
            ROS_ERROR_STREAM(msg.c_str());
            return;
        }
        ROS_INFO("Loading the rosbag %s", bag_path.c_str());
        
        rosbag::Bag bag;
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (rosbag::BagException &e) {
            ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
            return;
        }

        std::vector<string> lidar_topic_vec = {lidar_topic};
        rosbag::View view(bag, rosbag::TopicQuery(lidar_topic_vec));

        for (const rosbag::MessageInstance &m : view) 
        {
            // Determine if the message is a Livox custom message
            
            auto livox_custom_msg = m.instantiate<livox_ros_driver::CustomMsg>();
            if (livox_custom_msg) 
            {
                // Handle Livox custom message
                cloud_input_->reserve(livox_custom_msg->point_num);
                for (uint i = 0; i < livox_custom_msg->point_num; ++i) 
                {
                    pcl::PointXYZ p;
                    p.x = livox_custom_msg->points[i].x;
                    p.y = livox_custom_msg->points[i].y;
                    p.z = livox_custom_msg->points[i].z;
                    cloud_input_->points.push_back(p);
                }
            }
            else 
            {
                // Handle PCL format (Livox and Mechanical LiDAR)
                auto pcl_msg = m.instantiate<sensor_msgs::PointCloud2>();
                pcl::PointCloud<pcl::PointXYZ> temp_cloud;
                pcl::fromROSMsg(*pcl_msg, temp_cloud);
                *cloud_input_ += temp_cloud;
            } 
        }
        ROS_INFO("Loaded %ld points from the rosbag.", cloud_input_->size()); 
    }
};

typedef std::shared_ptr<DataPreprocess> DataPreprocessPtr;

#endif