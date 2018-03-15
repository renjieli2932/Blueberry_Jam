#!/usr/bin/env python
'''
Robot Manipulator Project "Blueberry Jam"
3/14/2018 Junji Hanatani

This code detects square, triangle, and circle objects in the input image.
Return detection marks on the original image and pixel positions of each object.

Required packages :numpy,cv2

Note: OpenCV version must be at least 3.x.x.
For version 2.x.x, the "findContours" returns two values (contours, hierarchy)
while it returns three (image, contours, hierarchy) for version 3.x.x.
'''
import cv2
import numpy as np

class ObjectDetection():
    '''
    Detect simple shaped objects (Circle, Triangle, Square)
    1)Split image into r,g,b and binarize using several threshould value.
    2)For each image, draw contours and extract objects.
    3)Filtering based on
        - area size
        - aspect ratio
        - convex hull
    4)Pick each shape based on the criteria bellow.
        -square: Object consists of 4 lines, angles are approximately 90deg
        -triangle: Object consists of 3 lines, angles are approximately 60deg
        -circle: Object area is nearly equal to the area of minimum enclosing circle.

    Reference:
    https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    https://docs.opencv.org/3.1.0/d1/d32/tutorial_py_contour_properties.html
    https://www.programcreek.com/python/example/86843/cv2.contourArea : Example 6
    https://github.com/opencv/opencv/tree/master/samples/python
    '''
    def __init__(self):
        # initialize
        self.circles = []
        self.squares = []
        self.triangles = []
        self.cnt_all = []

        # parameter setting
        self.area_max = 0.03
        self.area_min = 0.005
        self.aspect_ratio_max = 1.5
        self.aspect_ratio_min = 0.75
        self.angle_err = 10 * np.pi / 180
        self.circle_err = 0.85

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

    def sorter(self,cnt,n):
        norm = [np.linalg.norm(cnt[i]) for i in range(n)]
        ind = np.argsort(norm)
        return cnt[ind]

    def find_squares(self, cnt):
        if len(cnt) == 4: #and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([self.angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            min_cos = np.min([self.angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            if min_cos > -self.angle_err and max_cos < self.angle_err:
                cnt = self.sorter(cnt,4)
                tmp = cnt[3].copy()
                cnt[3] = cnt[2].copy()
                cnt[2] = tmp.copy()
                self.squares.append(cnt)
                self.cnt_all.append(cnt)

    def find_triangles(self,cnt):
        if len(cnt) == 3:
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([self.angle_cos( cnt[i], cnt[(i+1) % 3], cnt[(i+2) % 3] ) for i in range(3)])
            min_cos = np.min([self.angle_cos( cnt[i], cnt[(i+1) % 3], cnt[(i+2) % 3] ) for i in range(3)])
            if 0.5 - self.angle_err < min_cos and max_cos < 0.5 + self.angle_err:
                cnt = self.sorter(cnt,3)
                self.triangles.append(cnt)
                self.cnt_all.append(cnt)

    def find_circles(self,cnt):
        # roundness := Area of the object / Area of minEnclosingCircle
        area = cv2.contourArea(cnt)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        area_cnt = cv2.contourArea(cnt)
        area_circle = radius**2 * np.pi
        roundness = area_cnt/area_circle
        # Object filter based on solidity and roundness
        if roundness > self.circle_err:
            self.circles.append([int(x),int(y),radius])
            self.cnt_all.append(cnt)

    def draw_image(self,img):
        color = (0,0,255)
        thickness = 3
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_size = 1
        points = {'circle_center':(),'square_center':(),'triangle_center':(),
                    'circle_radius':0, 'square_vertices':[],'triangle_vertices':[]}

        # Draw a circle
        if self.circles != []:
            pts = np.mean(self.circles,axis = 0, dtype = np.int32)
            center = (pts[0],pts[1])
            radius = pts[2]
            cv2.circle(img, center, radius, color, thickness)
            cv2.putText(img,'circle'+str(center),center, font, font_size ,color)
            points['circle_center'] = center
            points['circle_radius'] = radius

        #Draw a rectangle
        if self.squares != []:
            pts = np.mean(self.squares,axis = 0, dtype = np.int32)
            pts = np.array(pts).reshape(-1,2)
            center = np.mean(pts,axis = 0, dtype = np.int32)
            center = (center[0],center[1])
            cv2.polylines(img, [pts], True, color, thickness)
            cv2.putText(img,'square'+str(center),center, font, font_size ,color)
            points['square_center'] = center
            points['square_vertices'] = pts

        #Draw a triangle
        if self.triangles != []:
            pts = np.mean(self.triangles,axis = 0, dtype = np.int32)
            pts = np.array(pts).reshape((-1,2))
            center = np.mean(pts,axis = 0, dtype = np.int32)
            center = (center[0],center[1])
            cv2.polylines(img, [pts], True, color, thickness)
            cv2.putText(img,'triangle'+str(center),center, font, font_size ,color)
            points['triangle_center'] = center
            points['triangle_vertices'] = pts

        return img, points

    def shape_recognition(self, img):
        '''
        Recognition for circle, square, and triangle
        '''
        img = cv2.GaussianBlur(img, (5, 5), 0)
        self.cnt_all = []
        self.squares = []
        self.triangles = []
        self.circles = []
        im_area = np.size(img,0)*np.size(img,1)
        for gray in cv2.split(img):
            for thrs in range(1, 255, 255/30):
                # Create Binary Image
                if thrs == 0:
                    bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                    bin = cv2.dilate(bin, None)
                else:
                    _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                # Draw contours
                _img_dummy, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    x,y,w,h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    cnt_len = cv2.arcLength(cnt, True)
                    # Object filter based on area and aspect ratio
                    if area > self.area_min*im_area and area < self.area_max*im_area and \
                        aspect_ratio > self.aspect_ratio_min and aspect_ratio < self.aspect_ratio_max:
                        cnt_rough = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                        cnt_fine = cv2.approxPolyDP(cnt, 0.005*cnt_len, True)
                        # Object filter based on convex hull
                        if cv2.isContourConvex(cnt_rough):
                            # Shape recognition
                            self.find_squares(cnt_rough)
                            self.find_triangles(cnt_rough)
                            self.find_circles(cnt_fine)

        #cv2.drawContours(img_res, self.cnt_all, -1, (255,255,255), 3)
        return self.draw_image(img)


if __name__ == '__main__':
    '''
    Realtime Video Capture by a built-in camera
    '''
    # Parameter setting
    ESC_KEY = 27     # Esc
    INTERVAL= 33     # Interval
    DEVICE_ID = 0    # Device ID
    WINDOW_NAME = "Circle, Square, Triangle Detection"

    # Create "VideoCapture" object
    cap = cv2.VideoCapture(DEVICE_ID)
    end_flag, org_frame = cap.read()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 15.0, (640,480))

    # Create video windows
    cv2.namedWindow(WINDOW_NAME)

    # ObjectDetection instance
    od = ObjectDetection()

    while end_flag == True:
        # Object Detection
        res_frame, points = od.shape_recognition(org_frame)
        # Display the frame
        cv2.imshow(WINDOW_NAME, res_frame)
        # Write the frame
        out.write(res_frame)
        # Press Esc to stop
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break
        # Read the next frame
        end_flag, org_frame = cap.read()

    # When everything done, close windows and release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
