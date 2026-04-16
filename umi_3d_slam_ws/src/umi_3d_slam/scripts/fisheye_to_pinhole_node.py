#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class FisheyeToPinholeNode:
    def __init__(self):
        self.frame_cnt = 0
        self.t_sum = 0.0

        # ---- params ----
        self.in_image_topic = rospy.get_param("~in_image_topic", "/left_camera/image")
        self.in_info_topic  = rospy.get_param("~in_info_topic",  "/left_camera/camera_info")  # 可选：用于继承 header/frame_id
        self.out_image_topic = rospy.get_param("~out_image_topic", "/left_camera/image_pinhole")
        self.out_info_topic  = rospy.get_param("~out_info_topic",  "/left_camera/camera_info_pinhole")

        # 你的脚本里的 K, D（默认值就是你上传脚本的示例）
        K_list = rospy.get_param("~K", [
            642.07805762, 0.0, 674.93483477,
            0.0, 642.27141104, 466.95742355,
            0.0, 0.0, 1.0
        ])
        D_list = rospy.get_param("~D", [-0.07606391, -0.01087557, 0.02406454, -0.01723208])

        self.K = np.array(K_list, dtype=np.float64).reshape(3, 3)
        self.D = np.array(D_list, dtype=np.float64).reshape(4, 1)

        # FAST-LIVO2 mode: newK = K
        self.newK = self.K.copy()
        self.R = np.eye(3, dtype=np.float64)

        # map cache
        self.map1 = None
        self.map2 = None
        self.dim = None  # (w,h)

        self.bridge = CvBridge()

        # pubs/subs
        self.pub_img = rospy.Publisher(self.out_image_topic, Image, queue_size=1)
        self.pub_info = rospy.Publisher(self.out_info_topic, CameraInfo, queue_size=1)

        self.last_in_info = None
        rospy.Subscriber(self.in_info_topic, CameraInfo, self._info_cb, queue_size=1)

        rospy.Subscriber(self.in_image_topic, Image, self._img_cb, queue_size=1, buff_size=2**24)

        rospy.loginfo("fisheye_to_pinhole_node ready.")
        rospy.loginfo("Sub: %s   Pub: %s", self.in_image_topic, self.out_image_topic)

    def _info_cb(self, msg: CameraInfo):
        self.last_in_info = msg

    def _build_maps_if_needed(self, w: int, h: int):
        dim = (w, h)
        if self.dim == dim and self.map1 is not None and self.map2 is not None:
            return

        self.dim = dim
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D,
            self.R,
            self.newK,
            dim,
            cv2.CV_16SC2
        )
        rospy.loginfo("Undistort maps built for DIM=%s (newK=K).", str(dim))

    def _make_pinhole_caminfo(self, header, w: int, h: int) -> CameraInfo:
        ci = CameraInfo()
        ci.header = header
        ci.width = w
        ci.height = h

        # 目标：输出是“针孔图像”，所以使用 plumb_bob + D=0
        ci.distortion_model = "plumb_bob"
        ci.D = [0.0, 0.0, 0.0, 0.0, 0.0]

        # K(new) = newK (这里等于 K)
        K = self.newK
        ci.K = [float(K[0, 0]), float(K[0, 1]), float(K[0, 2]),
                float(K[1, 0]), float(K[1, 1]), float(K[1, 2]),
                float(K[2, 0]), float(K[2, 1]), float(K[2, 2])]

        # R = I
        ci.R = [1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0]

        # P: [K | 0]
        ci.P = [float(K[0, 0]), float(K[0, 1]), float(K[0, 2]), 0.0,
                float(K[1, 0]), float(K[1, 1]), float(K[1, 2]), 0.0,
                0.0,          0.0,          1.0,          0.0]

        return ci

    def _img_cb(self, msg: Image):
        t0 = time.perf_counter()

        # decode
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        h, w = cv_img.shape[:2]
        self._build_maps_if_needed(w, h)

        und = cv2.remap(
            cv_img, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        out_msg = self.bridge.cv2_to_imgmsg(und, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)

        header = msg.header
        if self.last_in_info is not None:
            header = self.last_in_info.header
        self.pub_info.publish(self._make_pinhole_caminfo(header, w, h))

        # ---- timing ----
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0  # ms

        self.frame_cnt += 1
        self.t_sum += dt

        if self.frame_cnt % 30 == 0:
            rospy.loginfo(
                "[Fisheye->Pinhole] last: %.2f ms | avg(30): %.2f ms",
                dt, self.t_sum / self.frame_cnt
            )



def main():
    rospy.init_node("fisheye_to_pinhole_node", anonymous=False)
    _ = FisheyeToPinholeNode()
    rospy.spin()


if __name__ == "__main__":
    main()

