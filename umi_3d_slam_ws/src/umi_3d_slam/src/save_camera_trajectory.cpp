/*
 * save_camera_trajectory_cutstamp.cpp
 *
 * 20Hz timeline driven by CUT-STAMP topic (std_msgs/Time).
 * - Each cut-stamp -> one CSV row (frame_idx increments).
 * - LOST gate:
 *    lost==true  -> is_lost=true and pose = NaN
 *    lost==false -> if have odom, write pose; else NaN
 * - Keep NaN rows during IMU/EKF init (this is desired).
 *
 * Topics (defaults):
 *   - cut_stamp_topic: /livo/cut_stamp     (std_msgs/Time)  [from LIVMapper split]
 *   - odom_topic     : /camera_to_init    (nav_msgs/Odometry)
 *   - lost_topic     : /livo_frame_lost   (std_msgs/Bool)
 *
 * Params (~):
 *   - csv_path: camera_trajectory.csv
 *   - overwrite: true
 *   - use_relative_time: true
 *   - lost_default_if_unknown: true
 *   - state_default: 2
 *   - keyframe_default: false
 *   - flush_every_n: 50
 *   - write_have_odom_col: true
 *   - write_odom_age_col : true
 */

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Time.h>

#include <fstream>
#include <iomanip>
#include <string>
#include <mutex>
#include <limits>
#include <cmath>

class CameraTrajSaver
{
public:
  explicit CameraTrajSaver(ros::NodeHandle& nh)
  {
    nh.param<std::string>("cut_stamp_topic", cut_stamp_topic_, std::string("/livo/cut_stamp"));
    nh.param<std::string>("odom_topic",      odom_topic_,      std::string("/camera_to_init"));
    nh.param<std::string>("lost_topic",      lost_topic_,      std::string("/livo_frame_lost"));

    nh.param<std::string>("csv_path", csv_path_, std::string("camera_trajectory.csv"));
    nh.param<bool>("overwrite", overwrite_, true);
    nh.param<bool>("use_relative_time", use_relative_time_, true);

    nh.param<int>("state_default", state_default_, 2);
    nh.param<bool>("keyframe_default", keyframe_default_, false);

    nh.param<bool>("lost_default_if_unknown", lost_default_if_unknown_, true);

    nh.param<int>("flush_every_n", flush_every_n_, 50);
    if (flush_every_n_ <= 0) flush_every_n_ = 50;

    nh.param<bool>("write_have_odom_col", write_have_odom_col_, true);
    nh.param<bool>("write_odom_age_col",  write_odom_age_col_,  true);

    std::ios_base::openmode mode = std::ios::out;
    if (!overwrite_) mode |= std::ios::app;

    ofs_.open(csv_path_, mode);
    if (!ofs_.is_open())
    {
      ROS_FATAL_STREAM("Failed to open csv_path: " << csv_path_);
      ros::shutdown();
      return;
    }

    if (overwrite_)
    {
      ofs_ << "frame_idx,timestamp,state,is_lost,is_keyframe";
      if (write_have_odom_col_) ofs_ << ",have_odom";
      if (write_odom_age_col_)  ofs_ << ",odom_age";
      ofs_ << ",x,y,z,q_x,q_y,q_z,q_w\n";
      ofs_.flush();
    }

    cut_sub_  = nh.subscribe(cut_stamp_topic_, 2000, &CameraTrajSaver::cutStampCb, this);
    odom_sub_ = nh.subscribe(odom_topic_,      2000, &CameraTrajSaver::odomCb,     this);
    lost_sub_ = nh.subscribe(lost_topic_,      2000, &CameraTrajSaver::lostCb,     this);

    ROS_INFO_STREAM("Trajectory saver (CUT-STAMP driven):"
                    << " cut_stamp_topic=" << cut_stamp_topic_
                    << ", odom_topic=" << odom_topic_
                    << ", lost_topic=" << lost_topic_
                    << ", csv_path=" << csv_path_
                    << ", overwrite=" << (overwrite_ ? "true" : "false")
                    << ", use_relative_time=" << (use_relative_time_ ? "true" : "false")
                    << ", lost_default_if_unknown=" << (lost_default_if_unknown_ ? "true" : "false")
                    << ", flush_every_n=" << flush_every_n_
                    << ", write_have_odom_col=" << (write_have_odom_col_ ? "true" : "false")
                    << ", write_odom_age_col="  << (write_odom_age_col_  ? "true" : "false"));
  }

  ~CameraTrajSaver()
  {
    if (ofs_.is_open()) ofs_.close();
  }

private:
  void odomCb(const nav_msgs::OdometryConstPtr& msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    last_odom_ = *msg;
    last_odom_stamp_ = msg->header.stamp;
    have_odom_ = true;
  }

  void lostCb(const std_msgs::BoolConstPtr& msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    last_lost_ = msg->data;
    have_lost_ = true;
  }

  void cutStampCb(const std_msgs::TimeConstPtr& msg)
  {
    const ros::Time t_cut = msg->data;

    if (!t0_set_)
    {
      t0_ = t_cut;
      t0_set_ = true;
    }

    double t = t_cut.toSec();
    if (use_relative_time_) t = (t_cut - t0_).toSec();

    const int  state       = state_default_;
    const bool is_keyframe = keyframe_default_;

    bool is_lost = lost_default_if_unknown_;
    bool have_odom_local = false;
    bool use_pose = false;

    nav_msgs::Odometry odom_copy;
    ros::Time odom_stamp_copy(0);

    {
      std::lock_guard<std::mutex> lk(mtx_);

      if (have_lost_) is_lost = last_lost_;
      have_odom_local = have_odom_;

      if (!is_lost && have_odom_)
      {
        odom_copy = last_odom_;
        odom_stamp_copy = last_odom_stamp_;
        use_pose = true;
      }
    }

    double x  = std::numeric_limits<double>::quiet_NaN();
    double y  = std::numeric_limits<double>::quiet_NaN();
    double z  = std::numeric_limits<double>::quiet_NaN();
    double qx = std::numeric_limits<double>::quiet_NaN();
    double qy = std::numeric_limits<double>::quiet_NaN();
    double qz = std::numeric_limits<double>::quiet_NaN();
    double qw = std::numeric_limits<double>::quiet_NaN();

    if (use_pose)
    {
      const auto& p = odom_copy.pose.pose.position;
      const auto& q = odom_copy.pose.pose.orientation;
      x = p.x;  y = p.y;  z = p.z;
      qx = q.x; qy = q.y; qz = q.z; qw = q.w;
    }

    // odom_age: how stale is odom w.r.t this cut timestamp (seconds)
    double odom_age = std::numeric_limits<double>::quiet_NaN();
    if (write_odom_age_col_ && have_odom_local && odom_stamp_copy.toSec() > 0.0)
    {
      odom_age = (t_cut - odom_stamp_copy).toSec();
    }

    ofs_ << frame_idx_ << ","
         << std::fixed << std::setprecision(6) << t << ","
         << state << ","
         << (is_lost ? "true" : "false") << ","
         << (is_keyframe ? "true" : "false");

    if (write_have_odom_col_)
      ofs_ << "," << (have_odom_local ? "true" : "false");

    if (write_odom_age_col_)
      ofs_ << "," << std::setprecision(6) << odom_age;

    ofs_ << std::setprecision(9)
         << "," << x << "," << y << "," << z
         << "," << qx << "," << qy << "," << qz << "," << qw
         << "\n";

    if ((frame_idx_ % static_cast<uint64_t>(flush_every_n_)) == 0)
      ofs_.flush();

    frame_idx_++;
  }

private:
  std::string cut_stamp_topic_;
  std::string odom_topic_;
  std::string lost_topic_;

  std::ofstream ofs_;
  std::string csv_path_;
  bool overwrite_{true};
  bool use_relative_time_{true};

  int  state_default_{2};
  bool keyframe_default_{false};

  bool t0_set_{false};
  ros::Time t0_;

  uint64_t frame_idx_{0};

  std::mutex mtx_;
  nav_msgs::Odometry last_odom_;
  ros::Time last_odom_stamp_{0};
  bool have_odom_{false};

  bool last_lost_{true};
  bool have_lost_{false};

  bool lost_default_if_unknown_{true};
  int  flush_every_n_{50};
  bool write_have_odom_col_{true};
  bool write_odom_age_col_{true};

  ros::Subscriber cut_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber lost_sub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "save_camera_trajectory");
  ros::NodeHandle nh("~");
  CameraTrajSaver saver(nh);
  ros::spin();
  return 0;
}
