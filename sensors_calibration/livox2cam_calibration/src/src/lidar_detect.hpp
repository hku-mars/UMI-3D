/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_;

    // 存储中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

public:
    ros::Publisher filtered_pub_;
    ros::Publisher plane_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher edge_pub_;
    ros::Publisher center_z0_pub_;
    ros::Publisher center_pub_;

    LidarDetect(ros::NodeHandle &nh, Params &params)
        : filtered_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          plane_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;

        filtered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = nh.advertise<sensor_msgs::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = nh.advertise<sensor_msgs::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_cloud", 10);
    }

    // void detect_lidar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    // {
    //     // 1. X、Y、Z方向滤波
    //     filtered_cloud_->reserve(cloud->size());

    //     pcl::PassThrough<pcl::PointXYZ> pass_x;
    //     pass_x.setInputCloud(cloud);
    //     pass_x.setFilterFieldName("x");
    //     pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
    //     pass_x.filter(*filtered_cloud_);
    
    //     pcl::PassThrough<pcl::PointXYZ> pass_y;
    //     pass_y.setInputCloud(filtered_cloud_);
    //     pass_y.setFilterFieldName("y");
    //     pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
    //     pass_y.filter(*filtered_cloud_);
    
    //     pcl::PassThrough<pcl::PointXYZ> pass_z;
    //     pass_z.setInputCloud(filtered_cloud_);
    //     pass_z.setFilterFieldName("z");
    //     pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
    //     pass_z.filter(*filtered_cloud_);
    
    //     ROS_INFO("Filtered cloud size: %ld", filtered_cloud_->size());
        
    //     pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    //     voxel_filter.setInputCloud(filtered_cloud_);
    //     voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
    //     voxel_filter.filter(*filtered_cloud_);
    //     ROS_INFO("Filtered cloud size: %ld", filtered_cloud_->size());

    //     // 2. 平面分割
    //     plane_cloud_->reserve(filtered_cloud_->size());

    //     pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
    //     pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
    //     pcl::SACSegmentation<pcl::PointXYZ> plane_segmentation;
    //     plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
    //     plane_segmentation.setMethodType(pcl::SAC_RANSAC);
    //     plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
    //     plane_segmentation.setInputCloud(filtered_cloud_);
    //     plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
    //     pcl::ExtractIndices<pcl::PointXYZ> extract;
    //     extract.setInputCloud(filtered_cloud_);
    //     extract.setIndices(plane_inliers);
    //     extract.filter(*plane_cloud_);
    //     ROS_INFO("Plane cloud size: %ld", plane_cloud_->size());
    
    //     // 3. 平面点云对齐   
    //     aligned_cloud_->reserve(plane_cloud_->size());

    //     Eigen::Vector3d normal(plane_coefficients->values[0],
    //         plane_coefficients->values[1],
    //         plane_coefficients->values[2]);
    //     normal.normalize();
    //     Eigen::Vector3d z_axis(0, 0, 1);

    //     Eigen::Vector3d axis = normal.cross(z_axis);
    //     double angle = acos(normal.dot(z_axis));

    //     Eigen::AngleAxisd rotation(angle, axis);
    //     Eigen::Matrix3d R = rotation.toRotationMatrix();

    //     // 应用旋转矩阵，将平面对齐到 Z=0 平面
    //     float average_z = 0.0;
    //     int cnt = 0;
    //     for (const auto& pt : *plane_cloud_) {
    //         Eigen::Vector3d point(pt.x, pt.y, pt.z);
    //         Eigen::Vector3d aligned_point = R * point;
    //         aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
    //         average_z += aligned_point.z();
    //         cnt++;
    //     }
    //     average_z /= cnt;

    //     // 4. 提取边缘点
    //     edge_cloud_->reserve(aligned_cloud_->size());

    //     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    //     pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    //     normal_estimator.setInputCloud(aligned_cloud_);
    //     normal_estimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
    //     normal_estimator.compute(*normals);
    
    //     pcl::PointCloud<pcl::Boundary> boundaries;
    //     pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
    //     boundary_estimator.setInputCloud(aligned_cloud_);
    //     boundary_estimator.setInputNormals(normals);
    //     boundary_estimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
    //     boundary_estimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
    //     boundary_estimator.compute(boundaries);
    
    //     for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
    //         if (boundaries.points[i].boundary_point > 0) {
    //             edge_cloud_->push_back(aligned_cloud_->points[i]);
    //         }
    //     }
    //     ROS_INFO("Extracted %ld edge points.", edge_cloud_->size());

    //     // 5. 对边缘点进行聚类
    //     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    //     tree->setInputCloud(edge_cloud_);
    
    //     std::vector<pcl::PointIndices> cluster_indices;
    //     pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    //     ec.setClusterTolerance(0.02); // 设置聚类距离阈值
    //     ec.setMinClusterSize(50);     // 最小点数
    //     ec.setMaxClusterSize(1000);   // 最大点数
    //     ec.setSearchMethod(tree);
    //     ec.setInputCloud(edge_cloud_);
    //     ec.extract(cluster_indices);
    
    //     ROS_INFO("Number of edge clusters: %ld", cluster_indices.size());
    
    //     // 6. 对每个聚类进行圆拟合
    //     center_z0_cloud_->reserve(4);
    //     Eigen::Matrix3d R_inv = R.inverse();
    
    //     // 对每个聚类进行圆拟合
    //     for (size_t i = 0; i < cluster_indices.size(); ++i) 
    //     {
    //         pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
    //         for (const auto& idx : cluster_indices[i].indices) {
    //             cluster->push_back(edge_cloud_->points[idx]);
    //         }
    
    //         // 圆拟合
    //         pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    //         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //         pcl::SACSegmentation<pcl::PointXYZ> seg;
    //         seg.setOptimizeCoefficients(true);
    //         seg.setModelType(pcl::SACMODEL_CIRCLE2D);
    //         seg.setMethodType(pcl::SAC_RANSAC);
    //         seg.setDistanceThreshold(0.01); // 设置距离阈值
    //         seg.setMaxIterations(1000);     // 设置最大迭代次数
    //         seg.setInputCloud(cluster);
    //         seg.segment(*inliers, *coefficients);
    
    //         if (inliers->indices.size() > 0) 
    //         {
    //             // 计算拟合误差
    //             double error = 0.0;
    //             for (const auto& idx : inliers->indices) 
    //             {
    //                 double dx = cluster->points[idx].x - coefficients->values[0];
    //                 double dy = cluster->points[idx].y - coefficients->values[1];
    //                 double distance = sqrt(dx * dx + dy * dy) - circle_radius_; // 距离误差
    //                 error += abs(distance);
    //             }
    //             error /= inliers->indices.size();
    
    //             // 如果拟合误差较小，则认为是一个圆洞
    //             if (error < 0.025) 
    //             {
    //                 // 将恢复后的圆心坐标添加到点云中
    //                 pcl::PointXYZ center_point;
    //                 center_point.x = coefficients->values[0];
    //                 center_point.y = coefficients->values[1];
    //                 center_point.z = 0.0;
    //                 center_z0_cloud_->push_back(center_point);

    //                 // 将圆心坐标逆变换回原始坐标系
    //                 Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
    //                 Eigen::Vector3d original_point = R_inv * aligned_point;

    //                 pcl::PointXYZ center_point_origin;
    //                 center_point_origin.x = original_point.x();
    //                 center_point_origin.y = original_point.y();
    //                 center_point_origin.z = original_point.z();
    //                 center_cloud->points.push_back(center_point_origin);
    //             }
    //         }
    //     }
    // }
    void detect_lidar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // -----------------------------
        // 0) 清空缓存，防止累积
        // -----------------------------
        filtered_cloud_->clear();
        plane_cloud_->clear();
        aligned_cloud_->clear();
        edge_cloud_->clear();
        center_z0_cloud_->clear();
        center_cloud->clear();

        if (!cloud || cloud->empty()) {
            ROS_ERROR("[LidarDetect] Input cloud is empty!");
            return;
        }

        ROS_INFO("[LidarDetect] Input cloud size: %zu", cloud->size());

        // -----------------------------
        // 1) X/Y/Z PassThrough + Voxel
        // -----------------------------
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);

        pass.setFilterFieldName("x");
        pass.setFilterLimits(x_min_, x_max_);
        pass.filter(*filtered_cloud_);

        pass.setInputCloud(filtered_cloud_);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(y_min_, y_max_);
        pass.filter(*filtered_cloud_);

        pass.setInputCloud(filtered_cloud_);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_min_, z_max_);
        pass.filter(*filtered_cloud_);

        ROS_INFO("[LidarDetect] After ROI filter: %zu", filtered_cloud_->size());
        if (filtered_cloud_->size() < 1000) {
            ROS_WARN("[LidarDetect] ROI cloud too small. Check x/y/z limits.");
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(filtered_cloud_);
        voxel.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel.filter(*filtered_cloud_);
        ROS_INFO("[LidarDetect] After voxel(5mm): %zu", filtered_cloud_->size());

        // -----------------------------
        // 2) Plane segmentation (RANSAC)
        // -----------------------------
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coeff(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);

        pcl::SACSegmentation<pcl::PointXYZ> seg_plane;
        seg_plane.setOptimizeCoefficients(true);
        seg_plane.setModelType(pcl::SACMODEL_PLANE);
        seg_plane.setMethodType(pcl::SAC_RANSAC);
        seg_plane.setDistanceThreshold(0.01);
        seg_plane.setMaxIterations(1000);
        seg_plane.setInputCloud(filtered_cloud_);
        seg_plane.segment(*plane_inliers, *plane_coeff);

        if (plane_inliers->indices.empty() || plane_coeff->values.size() < 4) {
            ROS_ERROR("[LidarDetect] Plane segmentation failed. inliers=%zu, coeff_size=%zu",
                    plane_inliers->indices.size(), plane_coeff->values.size());
            return;
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("[LidarDetect] Plane cloud size: %zu (inliers=%zu)", plane_cloud_->size(), plane_inliers->indices.size());

        // -----------------------------
        // 3) Align plane to z=0 via rotation
        //     R * point  => aligned_point
        // -----------------------------
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coeff->values[0], plane_coeff->values[1], plane_coeff->values[2]);
        double n_norm = normal.norm();
        if (n_norm < 1e-12) {
            ROS_ERROR("[LidarDetect] Plane normal invalid (norm~0).");
            return;
        }
        normal /= n_norm;

        Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
        double dot = normal.dot(z_axis);
        dot = std::max(-1.0, std::min(1.0, dot));
        double angle = std::acos(dot);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double axis_norm = axis.norm();

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (axis_norm < 1e-10 || angle < 1e-10) {
            // normal already ~ z, no rotation needed
            R.setIdentity();
            ROS_INFO("[LidarDetect] Plane normal ~ z-axis. Use Identity rotation.");
        } else {
            axis /= axis_norm;
            Eigen::AngleAxisd aa(angle, axis);
            R = aa.toRotationMatrix();
            ROS_INFO("[LidarDetect] Align rotation: angle=%.6f rad, axis=(%.6f, %.6f, %.6f)",
                    angle, axis.x(), axis.y(), axis.z());
        }

        // 计算 average_z（用 aligned_point 的 z 的均值）
        double average_z = 0.0;
        size_t cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            Eigen::Vector3d ap = R * p;
            // 存入“强制 z=0”平面点（2D化）
            aligned_cloud_->push_back(pcl::PointXYZ(ap.x(), ap.y(), 0.0));
            average_z += ap.z();
            cnt++;
        }
        average_z /= std::max<size_t>(1, cnt);
        ROS_INFO("[LidarDetect] average_z (aligned frame) = %.6f, aligned_cloud=%zu",
                average_z, aligned_cloud_->size());

        // -----------------------------
        // 4) Boundary / edge extraction
        // -----------------------------
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud(aligned_cloud_);
        ne.setRadiusSearch(0.03);
        ne.compute(*normals);

        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> be;
        be.setInputCloud(aligned_cloud_);
        be.setInputNormals(normals);
        be.setRadiusSearch(0.03);
        be.setAngleThreshold(M_PI / 4);
        be.compute(boundaries);

        if (boundaries.size() != aligned_cloud_->size()) {
            ROS_WARN("[LidarDetect] Boundary size mismatch: boundaries=%zu, aligned=%zu",
                    boundaries.size(), aligned_cloud_->size());
        }

        for (size_t i = 0; i < aligned_cloud_->size() && i < boundaries.size(); ++i) {
            if (boundaries.points[i].boundary_point > 0) {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        ROS_INFO("[LidarDetect] Extracted edge points: %zu", edge_cloud_->size());

        if (edge_cloud_->size() < 200) {
            ROS_WARN("[LidarDetect] Edge points too few; circle fitting may fail.");
        }

        // -----------------------------
        // 5) Euclidean clustering on edge points
        // -----------------------------
        std::vector<pcl::PointIndices> cluster_indices;
        if (!edge_cloud_->empty()) {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(edge_cloud_);

            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.02);
            ec.setMinClusterSize(50);
            ec.setMaxClusterSize(5000); // 放宽，避免截断
            ec.setSearchMethod(tree);
            ec.setInputCloud(edge_cloud_);
            ec.extract(cluster_indices);
        }
        ROS_INFO("[LidarDetect] Number of edge clusters: %zu", cluster_indices.size());

        // -----------------------------
        // 6) Circle2D fit for each cluster
        //     Use r_fit residual + radius consistency separately
        // -----------------------------
        center_z0_cloud_->reserve(4);

        Eigen::Matrix3d R_inv = R.inverse();

        // 这些阈值你后续可以参数化；先用更“能出点”的宽松值排查
        const double r_tol = 0.04;              // 半径容差：4cm（先排查，稳定后可收紧）
        const double max_mean_res = 0.03;       // 平均残差：3cm（先排查）
        const double ransac_dist_th = 0.02;     // RANSAC 距离阈值：2cm（比你原来的 1cm 宽松）

        size_t accepted = 0;

        for (size_t i = 0; i < cluster_indices.size(); ++i)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            cluster->reserve(cluster_indices[i].indices.size());
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }

            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CIRCLE2D);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(ransac_dist_th);
            seg.setMaxIterations(2000);
            seg.setInputCloud(cluster);
            seg.segment(*inliers, *coeff);

            // coeff: [cx, cy, r]
            double cx = 0, cy = 0, r_fit = -1;
            if (coeff->values.size() >= 3) {
                cx = coeff->values[0];
                cy = coeff->values[1];
                r_fit = coeff->values[2];
            }

            ROS_INFO("[CircleFit] cluster %zu: pts=%zu inliers=%zu coeff_size=%zu cx=%.4f cy=%.4f r_fit=%.4f",
                    i, cluster->size(), inliers->indices.size(), coeff->values.size(), cx, cy, r_fit);

            if (inliers->indices.empty() || coeff->values.size() < 3 || !(r_fit > 0)) {
                ROS_WARN("[CircleFit] cluster %zu rejected: no valid circle model.", i);
                continue;
            }

            // 半径一致性检查
            double rad_err = std::abs(r_fit - circle_radius_);
            if (rad_err > r_tol) {
                ROS_WARN("[CircleFit] cluster %zu rejected by radius: r_fit=%.4f expected=%.4f |err|=%.4f tol=%.4f",
                        i, r_fit, circle_radius_, rad_err, r_tol);
                continue;
            }

            // 残差：mean |dist - r_fit|
            double mean_abs_res = 0.0;
            for (const auto& idx : inliers->indices) {
                double dx = cluster->points[idx].x - cx;
                double dy = cluster->points[idx].y - cy;
                double dist = std::sqrt(dx*dx + dy*dy);
                mean_abs_res += std::abs(dist - r_fit);
            }
            mean_abs_res /= std::max<size_t>(1, inliers->indices.size());

            ROS_INFO("[CircleFit] cluster %zu: mean_abs_res=%.4f (th=%.4f), rad_err=%.4f",
                    i, mean_abs_res, max_mean_res, rad_err);

            if (mean_abs_res > max_mean_res) {
                ROS_WARN("[CircleFit] cluster %zu rejected by residual: mean=%.4f th=%.4f",
                        i, mean_abs_res, max_mean_res);
                continue;
            }

            // 通过：记录中心点（aligned frame, z=0）
            pcl::PointXYZ center_point;
            center_point.x = cx;
            center_point.y = cy;
            center_point.z = 0.0;
            center_z0_cloud_->push_back(center_point);

            // 逆变换回原始坐标系：先补回 z（用 average_z），再乘 R_inv
            Eigen::Vector3d aligned_center(center_point.x, center_point.y, average_z);
            Eigen::Vector3d original_center = R_inv * aligned_center;

            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_center.x();
            center_point_origin.y = original_center.y();
            center_point_origin.z = original_center.z();
            center_cloud->points.push_back(center_point_origin);

            accepted++;
            ROS_INFO("[CircleFit] cluster %zu ACCEPTED -> center_origin=(%.4f, %.4f, %.4f)",
                    i, center_point_origin.x, center_point_origin.y, center_point_origin.z);
        }

        ROS_INFO("[LidarDetect] Accepted centers: %zu (expected 4). center_cloud=%zu, center_z0=%zu",
                accepted, center_cloud->size(), center_z0_cloud_->size());
    }


    // 获取中间结果的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif