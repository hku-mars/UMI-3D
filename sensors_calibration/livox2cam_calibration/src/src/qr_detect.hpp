/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>
Modified: fisheye PnP + fisheye projection for RAW fisheye images
- Use aruco detectMarkers to get pixel corners (model-free)
- Use fisheye::undistortPoints + solvePnP to estimate pose
  * DEFAULT: undistort to pixel plane with a scaled virtual camera K (demo-style, more stable near borders)
  * DEBUG: also compute normalized-plane solution (K=I) for comparison
- Use fisheye::projectPoints for debug drawing on RAW fisheye image

NOTE:
Params use fisheye distortion coefficients directly:
  D_fisheye = [k1, k2, k3, k4]^T
*/

#ifndef QR_DETECT_HPP
#define QR_DETECT_HPP

#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>

#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

#include "common_lib.h"

class QRDetect 
{
private:
  double marker_size_, delta_width_qr_center_, delta_height_qr_center_;
  double delta_width_circles_, delta_height_circles_;
  int min_detected_markers_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;

  // demo-style scale: enlarge virtual undistorted pixel plane
  static constexpr int FISHEYE_PNP_SCALE = 4;

  // helper: board id -> index in boardCorners
  inline int boardIndexFromId(int id) const {
    // boardIds = {1,2,4,3} mapped to indices {0,1,2,3}
    if (id == 1) return 0;
    if (id == 2) return 1;
    if (id == 4) return 2;
    if (id == 3) return 3;
    return -1;
  }

  static inline double nowSec(const ros::Time& t){
    return (double)t.sec + 1e-9 * (double)t.nsec;
  }

  // fisheye pixel -> normalized (x,y) plane (z=1). Output is normalized if P is omitted/identity.
  static void undistortPointsFisheyeNormalized(
      const std::vector<cv::Point2f>& px,
      std::vector<cv::Point2f>& norm,
      const cv::Mat& K,
      const cv::Mat& D)
  {
    // Output Nx1x2 normalized (x,y). Using default R=I, P=I.
    cv::fisheye::undistortPoints(px, norm, K, D);
  }

  // demo-style: fisheye pixel -> undistorted PIXEL on a virtual pinhole camera plane (newK),
  // with enlarged image size (scale*W, scale*H) and principal point at the center.
  static void undistortPointsFisheyeToPixel(
      const std::vector<cv::Point2f>& px,
      std::vector<cv::Point2f>& undist_px,
      const cv::Mat& K,
      const cv::Mat& D,
      int img_w,
      int img_h,
      int scale,
      cv::Mat* out_newK,
      cv::Size* out_newSize)
  {
    cv::Size new_size(img_w * scale, img_h * scale);

    cv::Mat newK = K.clone();
    newK.at<double>(0,2) = 0.5 * (double)new_size.width;
    newK.at<double>(1,2) = 0.5 * (double)new_size.height;

    // Output as pixel points on the virtual camera plane defined by newK.
    cv::fisheye::undistortPoints(px, undist_px, K, D, cv::Matx33d::eye(), newK);

    if (out_newK) *out_newK = newK;
    if (out_newSize) *out_newSize = new_size;
  }

  static inline bool inBounds(const cv::Point2f& p, int W, int H) {
    return (p.x >= 0.f && p.y >= 0.f && p.x < (float)W && p.y < (float)H);
  }

  // fisheye projection for debug (raw fisheye image domain)
  cv::Point2f projectPointFisheye(const cv::Point3f& pt_cam) const {
    std::vector<cv::Point3f> obj(1, pt_cam);
    std::vector<cv::Point2f> img(1);
    cv::Mat rvec = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1,CV_64F);
    cv::fisheye::projectPoints(obj, img, rvec, tvec, cameraMatrix_, distCoeffs_);
    return img[0];
  }

public:
  ros::Publisher qr_pub_;
  cv::Mat imageCopy_;
  cv::Mat cameraMatrix_;   // 3x3, double
  cv::Mat distCoeffs_;     // 4x1, double (fisheye k1..k4)

  QRDetect(ros::NodeHandle &nh, Params& params) 
  {
    marker_size_ = params.marker_size;
    delta_width_qr_center_ = params.delta_width_qr_center;
    delta_height_qr_center_ = params.delta_height_qr_center;
    delta_width_circles_ = params.delta_width_circles;
    delta_height_circles_ = params.delta_height_circles;
    min_detected_markers_ = params.min_detected_markers;

    // Intrinsics (RAW fisheye)
    cameraMatrix_ = (cv::Mat_<double>(3, 3) << params.fx, 0, params.cx,
                                               0, params.fy, params.cy,
                                               0,         0,        1);

    distCoeffs_ = (cv::Mat_<double>(4, 1) << params.k1, params.k2, params.k3, params.k4);

    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    qr_pub_ = nh.advertise<sensor_msgs::PointCloud2>("qr_cloud", 1);

    ROS_INFO("[QRDetect] Using FISHEYE model. D=[k1 k2 k3 k4]=[%+.6f %+.6f %+.6f %+.6f]  PnPScale=%d",
         params.k1, params.k2, params.k3, params.k4, FISHEYE_PNP_SCALE);
  }

  void comb(int N, int K, std::vector<std::vector<int>> &groups) {
    int upper_factorial = 1;
    int lower_factorial = 1;

    for (int i = 0; i < K; i++) {
      upper_factorial *= (N - i);
      lower_factorial *= (K - i);
    }
    int n_permutations = upper_factorial / lower_factorial;

    if (DEBUG)
      cout << N << " centers found. Iterating over " << n_permutations
           << " possible sets of candidates" << endl;

    std::string bitmask(K, 1);
    bitmask.resize(N, 0);

    do {
      std::vector<int> group;
      for (int i = 0; i < N; ++i) {
        if (bitmask[i]) group.push_back(i);
      }
      groups.push_back(group);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

    assert((int)groups.size() == n_permutations);
  }

  void detect_qr(cv::Mat &image, pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud) 
  {
    image.copyTo(imageCopy_);
    centers_cloud->clear();

    // Build board geometry (same as original)
    std::vector<std::vector<cv::Point3f>> boardCorners;
    std::vector<cv::Point3f> boardCircleCenters;

    float width = (float)delta_width_qr_center_;
    float height = (float)delta_height_qr_center_;
    float circle_width = (float)delta_width_circles_ / 2.f;
    float circle_height = (float)delta_height_circles_ / 2.f;

    boardCorners.resize(4);

    for (int i = 0; i < 4; ++i) {
      int x_qr_center = ((i % 3) == 0) ? -1 : 1;
      int y_qr_center = (i < 2) ? 1 : -1;

      float x_center = x_qr_center * width;
      float y_center = y_qr_center * height;

      cv::Point3f circleCenter3d(x_qr_center * circle_width,
                                 y_qr_center * circle_height, 0);
      boardCircleCenters.push_back(circleCenter3d);

      for (int j = 0; j < 4; ++j) {
        int x_qr = ((j % 3) == 0) ? -1 : 1;
        int y_qr = (j < 2) ? 1 : -1;

        cv::Point3f pt3d(x_center + x_qr * (float)marker_size_ / 2.f,
                         y_center + y_qr * (float)marker_size_ / 2.f, 0);
        boardCorners[i].push_back(pt3d);
      }
    }

    // Detect markers (pixel corners)
    cv::Ptr<cv::aruco::DetectorParameters> parameters =
        cv::aruco::DetectorParameters::create();
#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
    parameters->doCornerRefinement = true;
#else
    parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(image, dictionary_, corners, ids, parameters);

    if (ids.size() > 0) {
      cv::aruco::drawDetectedMarkers(imageCopy_, corners, ids);
    }

    if (!(ids.size() >= (size_t)min_detected_markers_ && ids.size() <= TARGET_NUM_CIRCLES)) {
      ROS_WARN("%lu marker(s) found, %d expected(min). Skipping frame...",
               ids.size(), min_detected_markers_);
      return;
    }

    // Assemble object/image correspondences for PnP (board markers -> board corners)
    std::vector<cv::Point3f> obj_all;
    std::vector<cv::Point2f> img_all_raw;
    obj_all.reserve(ids.size() * 4);
    img_all_raw.reserve(ids.size() * 4);

    for (int i = 0; i < (int)ids.size(); ++i) {
      int bid = boardIndexFromId(ids[i]);
      if (bid < 0) continue;
      for (int k = 0; k < 4; ++k) {
        obj_all.push_back(boardCorners[bid][k]);
        img_all_raw.push_back(corners[i][k]);
      }
    }

    if (obj_all.size() < 4) {
      ROS_WARN("[FisheyePnP] Not enough points for solvePnP: %zu", obj_all.size());
      return;
    }

    // ======== NEW DEFAULT: demo-style pixel-plane undistort + solvePnP(newK, dist=0) ========
    std::vector<cv::Point2f> img_undist_px;
    cv::Mat newK;
    cv::Size newSize;
    undistortPointsFisheyeToPixel(img_all_raw, img_undist_px,
                                  cameraMatrix_, distCoeffs_,
                                  image.cols, image.rows,
                                  FISHEYE_PNP_SCALE,
                                  &newK, &newSize);

    // Filter out-of-bounds points in the enlarged virtual image (important for stability)
    std::vector<cv::Point3f> obj_f;
    std::vector<cv::Point2f> img_f;
    obj_f.reserve(obj_all.size());
    img_f.reserve(img_undist_px.size());

    for (size_t i = 0; i < img_undist_px.size(); ++i) {
      if (inBounds(img_undist_px[i], newSize.width, newSize.height)) {
        obj_f.push_back(obj_all[i]);
        img_f.push_back(img_undist_px[i]);
      }
    }

    if (obj_f.size() < 4) {
      ROS_WARN("[FisheyePnP-Pixel] Too many points out of bounds after undistort: kept=%zu/%zu",
               obj_f.size(), obj_all.size());
      return;
    }

    cv::Mat rvec_pnp, tvec_pnp;
    cv::Mat dist0_5 = cv::Mat::zeros(5,1,CV_64F); // no distortion after undistortion

    bool ok = cv::solvePnP(obj_f, img_f, newK, dist0_5,
                           rvec_pnp, tvec_pnp, false, cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
      ROS_WARN("[FisheyePnP-Pixel] solvePnP failed.");
      return;
    }

    // ======== DEBUG: also compute normalized-plane solution for comparison ========
    if (DEBUG) {
      std::vector<cv::Point2f> img_norm;
      undistortPointsFisheyeNormalized(img_all_raw, img_norm, cameraMatrix_, distCoeffs_);
      cv::Mat rvec_n, tvec_n;
      cv::Mat K_I = cv::Mat::eye(3,3,CV_64F);
      cv::Mat dist0_4 = cv::Mat::zeros(4,1,CV_64F);

      bool ok_n = cv::solvePnP(obj_all, img_norm, K_I, dist0_4,
                               rvec_n, tvec_n, false, cv::SOLVEPNP_ITERATIVE);

      ROS_INFO("[FisheyePnP] used_points(raw=%zu, kept_px=%zu)  "
               "tz_pix=%.4f  %s tz_norm=%.4f  scale=%d",
               obj_all.size(), obj_f.size(),
               tvec_pnp.at<double>(2),
               ok_n ? "|" : "(norm_fail)|",
               ok_n ? tvec_n.at<double>(2) : 0.0,
               FISHEYE_PNP_SCALE);

      ROS_INFO("[FisheyePnP] rvec_pix=[%+.4f %+.4f %+.4f] tvec_pix=[%+.4f %+.4f %+.4f]",
               rvec_pnp.at<double>(0), rvec_pnp.at<double>(1), rvec_pnp.at<double>(2),
               tvec_pnp.at<double>(0), tvec_pnp.at<double>(1), tvec_pnp.at<double>(2));
    }

    // Build board_transform (board -> camera)
    cv::Mat R;
    cv::Rodrigues(rvec_pnp, R); // 3x3
    cv::Mat board_transform = cv::Mat::eye(3, 4, CV_64F);
    R.copyTo(board_transform(cv::Rect(0,0,3,3)));
    tvec_pnp.copyTo(board_transform(cv::Rect(3,0,1,3)));

    // Compute 3D circle centers in camera frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr candidates_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    candidates_cloud->reserve(boardCircleCenters.size());

    for (int i = 0; i < (int)boardCircleCenters.size(); ++i) {
      cv::Mat X = (cv::Mat_<double>(4,1) << boardCircleCenters[i].x,
                                            boardCircleCenters[i].y,
                                            boardCircleCenters[i].z,
                                            1.0);
      cv::Mat Y = board_transform * X; // 3x1 camera coords
      cv::Point3f center3d((float)Y.at<double>(0),
                           (float)Y.at<double>(1),
                           (float)Y.at<double>(2));

      // DEBUG draw center on RAW fisheye image
      if (DEBUG) {
        cv::Point2f uv = projectPointFisheye(center3d);
        cv::circle(imageCopy_, uv, 5, cv::Scalar(0,255,0), -1);
      }

      pcl::PointXYZ qr_center;
      qr_center.x = center3d.x;
      qr_center.y = center3d.y;
      qr_center.z = center3d.z;
      candidates_cloud->push_back(qr_center);
    }

    // Same geometric consistency check as original
    std::vector<std::vector<int>> groups;
    comb((int)candidates_cloud->size(), TARGET_NUM_CIRCLES, groups);

    std::vector<double> groups_scores(groups.size(), -1.0);

    for (int i = 0; i < (int)groups.size(); ++i) {
      std::vector<pcl::PointXYZ> candidates;
      for (int j = 0; j < (int)groups[i].size(); ++j) {
        candidates.push_back(candidates_cloud->at(groups[i][j]));
      }

      Square square_candidate(candidates, delta_width_circles_, delta_height_circles_);
      groups_scores[i] = square_candidate.is_valid() ? 1.0 : -1.0;
    }

    int best_candidate_idx = -1;
    double best_candidate_score = -1;

    for (int i = 0; i < (int)groups.size(); ++i) {
      if (best_candidate_score == 1 && groups_scores[i] == 1) {
        ROS_ERROR("[Mono-Fisheye] More than one candidate set fits target geometry. Check params.");
        return;
      }
      if (groups_scores[i] > best_candidate_score) {
        best_candidate_score = groups_scores[i];
        best_candidate_idx = i;
      }
    }

    if (best_candidate_idx == -1) {
      ROS_WARN("[Mono-Fisheye] No candidate set matches target geometry.");
      return;
    }

    for (int j = 0; j < (int)groups[best_candidate_idx].size(); ++j) {
      centers_cloud->push_back(candidates_cloud->at(groups[best_candidate_idx][j]));
    }

    if (DEBUG) {
      for (int i = 0; i < (int)centers_cloud->size(); i++) {
        cv::Point3f pt((float)centers_cloud->at(i).x,
                       (float)centers_cloud->at(i).y,
                       (float)centers_cloud->at(i).z);
        cv::Point2f uv = projectPointFisheye(pt);
        cv::circle(imageCopy_, uv, 2, cv::Scalar(255,0,255), -1);
      }
      ROS_INFO("[Mono-Fisheye] centers_cloud size=%zu", centers_cloud->size());
    }

    // Publish pointcloud message (optional)
    // (原代码中 publish 在 main 循环里完成，这里不动)
  }
};

typedef std::shared_ptr<QRDetect> QRDetectPtr;

#endif
