# object_detection.py
This code subscribes an original image captured by a camera module and publishes object detection results. In this code, "ObjectDetection()" class is called from "simple_shape_recognition.py" (see below.) <br/>

Subscribe<br/>
Topic: /xxxxx <br/>
Message type: sensor_msgs/Image <br/>

Publish<br/>
Topic: /object_detect_image <br/>
Message type: sensor_msgs/Image <br/>

Required packages: rospy, numpy, cv_bridge

# simple_shape_recognition.py
This code detects square, triangle, and circle in the input image, and returns detection marks on the original image and pixel positions of each object.
1. Split image into r,g,b and binarize the image using several threshold values.
2. For each image, draw contours and extract objects.
3. Filter the objects based on
	* area size
	* aspect ratio
	* convex hull

4. Pick each shape based on the criteria bellow. <br/>
	* square: Object consists of 4 lines and the angles are approximately 90deg.
	* triangle: Object consists of 3 lines and the angles are approximately 60deg.
	* circle: Object area is nearly equal to the area of minimum enclosing circle.

Required packages :numpy,cv2 <br/>
Note: OpenCV version must be at least 3.x.x.

<br/>
Reference:
https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
https://docs.opencv.org/3.1.0/d1/d32/tutorial_py_contour_properties.html
https://www.programcreek.com/python/example/86843/cv2.contourArea(Example 6)
https://github.com/opencv/opencv/tree/master/samples/python
