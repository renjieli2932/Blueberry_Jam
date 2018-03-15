#!/usr/bin/env python
'''
Robot Manipulator Project "Blueberry Jam"
3/14/2018 Junji Hanatani

Subscribe an original image captured by a camera module and
Publish object detection results.

<Subscribe>
Topic Name: /cameras/left_hand_camera/image
Message type: sensor_msgs/Image

<Publish>
Topic Name: /detected_image
Message type: sensor_msgs/Image
Topic Name: /detected_object
Message type:  baxter_object_detection/Object

Required packages: rospy, numpy, cv_bridge
Note: "cv_bridge" converts ROS images into OpenCV images and vice versa.
Source: git https://github.com/ros-perception/vision_opencv.git
'''
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from baxter_object_detection.msg import Object
from geometry_msgs.msg import Point32
from simple_shape_recognition import ObjectDetection

def publish(img,points):

    # Publisher
    image_pub = rospy.Publisher("/detected_image",Image,queue_size=1)
    object_pub = rospy.Publisher("/detected_object",Object,queue_size=1)

    # Publish image
    image_pub.publish(img)

    # Create msg for center position
    msg = Object()
    msg.circle_center = points['circle_center']
    msg.triangle_center = points['triangle_center']
    msg.square_center = points['square_center']

    # Create msg for circle radius
    msg.circle_radius = points['circle_radius']

    # Create msg for triangle vertices
    point = Point32()
    msg.triangle_vertices = []
    for p in points['triangle_vertices']:
        point.x = p[0]
        point.y = p[1]
        msg.triangle_vertices.append(point)

    # Create msg for square vertices
    msg.square_vertices = []
    for p in points['triangle_vertices']:
        point.x = p[0]
        point.y = p[1]
        msg.triangle_vertices.append(point)

    # Publish object
    object_pub.publish(msg)

def callback(image):
    bridge = CvBridge()
    od = ObjectDetection()
    # Convert ROS image into OpenCV image
    cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
    # Object Detection
    cv_image, points = od.shape_recognition(cv_image)
    # Convert OpenCV image into ROS image
    ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    ros_image.header.frame_id = "object_detect_image"
    # Publish
    publish(ros_image, points)

if __name__ == '__main__':
    # Node definition
    rospy.init_node('object_detector', anonymous=True)
    # Subscribe Image from camera
    rospy.Subscriber("/cameras/left_hand_camera/image", Image, callback)
    #rospy.Subscriber("/usb_cam/image_raw", Image, callback)
    rospy.spin()
