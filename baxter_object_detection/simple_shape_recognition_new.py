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
import scipy.stats
import matplotlib.pyplot as plt

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
    def __init__(self,img):
        # initialize
        self.circles = []
        self.squares = []
        self.triangles = []
        self.im_x = 0
        self.im_y = 0

        # parameter setting
        self.area_max = 0.03
        self.area_min = 0.0005
        self.aspect_ratio_max = 1.5
        self.aspect_ratio_min = 0.75
        self.angle_err = 10 * np.pi / 180
        self.circle_err = 0.85
        self.thrs_list = [0,100,150,200]
        self.binsize = [5,5]
        self.im_x = np.size(img,0)
        self.im_y = np.size(img,1)
        self.bins = [int(self.im_x/self.binsize[0]),int(self.im_y/self.binsize[1])]
        #shape = self.bins.copy()
        #shape.append(3)
        #self. = np.zeros(shape)

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) )

    def sorter(self,cnt):
        p0 = cnt[np.argmin(cnt[:,1])]
        denom = [np.linalg.norm(pt-p0) for pt in cnt]
        numer = [pt[0] - p0[0] for pt in cnt]
        cos = [numer[i]/denom[i] if denom[i] !=0 else -1.0 for i in range(len(cnt))]
        cnt_sort = cnt[np.argsort(cos)]
        return cnt_sort

    def find_squares(self, cnt):
        cnt_square, cos_list = self.my_approxPoly(cnt, 4)
        approx_rate = cv2.contourArea(cnt_square)/cv2.contourArea(cnt)
        angle_err = np.abs(90 - np.arccos(cos_list)*180/np.pi)
        max_angle_err = np.max(angle_err)
        if approx_rate > 0.85 and max_angle_err < 10:
            cnt_square_sort = self.sorter(cnt_square)
            self.squares.append([cnt_square_sort, approx_rate, max_angle_err])

    def find_triangles(self,cnt):
        cnt_triangle, cos_list = self.my_approxPoly(cnt, 3)
        approx_rate = cv2.contourArea(cnt_triangle)/cv2.contourArea(cnt)
        angle_err = np.abs(60 - np.arccos(cos_list)*180/np.pi)
        max_angle_err = np.max(angle_err)
        if approx_rate > 0.85 and max_angle_err <10:
            cnt_triangle_sort = self.sorter(cnt_triangle)
            self.triangles.append([cnt_triangle_sort, approx_rate, max_angle_err])
            center = np.mean(cnt_triangle_sort, axis = 0, dtype = np.int32)

    def find_circles(self,cnt):
        # roundness := Area of the object / Area of minEnclosingCircle
        area = cv2.contourArea(cnt)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        area_cnt = cv2.contourArea(cnt)
        area_circle = radius**2 * np.pi
        roundness = area_cnt/area_circle
        if roundness > 0.85:
            self.circles.append([[int(x),int(y),radius], roundness])

    def binning2D(self,data):
        x = np.array(data)[:,0]
        y = np.array(data)[:,1]
        value = np.zeros(x.shape)
        x_range = [0,self.im_x]
        y_range = [0,self.im_y]
        res = scipy.stats.binned_statistic_2d(x,y,value, \
                statistic='count', bins=self.bins, range=[x_range,y_range])
        return res

    def draw_image(self,img):
        color = (0,0,255)
        thickness = 3
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_size = 1
        points = {'circle_center':(),'square_center':(),'triangle_center':(),
                    'circle_radius':0, 'square_vertices':[],'triangle_vertices':[]}

        img_chk = [np.zeros([self.im_x, self.im_y],dtype='uint8') for i in range(6)]
        img_chk[0] = self.check[0]
        cv2.drawContours(img_chk[1], self.check[1], -1, 255, 1)
        cv2.drawContours(img_chk[2], self.check[2], -1, 255, 1)

        # Draw a circle
        if self.circles != []:
            self.circles = np.array(self.circles)
            ind = np.argsort(self.circles[:,1])
            self.circles = self.circles[ind,:]

            for i, segments in enumerate(self.circles):
                pts = segments[0]
                center = (pts[0],pts[1])
                radius = int(pts[2])
                score = segments[1]
                color_mono = int(score*255)
                cv2.circle(img_chk[5], center, radius, color_mono, 1)

                if i==0:
                    cv2.circle(img, center, radius, color, 3)
                    cv2.putText(img,'circle'+str(center), center, font, font_size ,color)
                    points['circle_center'] = center
                    points['circle_radius'] = radius

        # Draw a square
        if self.squares != []:
            self.squares = np.array(self.squares)
            ind = np.argsort(self.squares[:,1])
            self.squares = self.squares[ind,:]

            for i, segments in enumerate(self.squares):
                pts = segments[0]
                score = segments[1]
                err = segments[2]
                color_mono = int(score*255)
                cv2.polylines(img_chk[4], [pts], True, color_mono, 1)

                if i==0:
                    center = np.mean(pts,axis = 0, dtype = np.int32)
                    center = (center[0],center[1])
                    cv2.polylines(img, [pts], True, color, thickness)
                    cv2.putText(img,'square'+str(center),center, font, font_size ,color)
                    points['square_center'] = center
                    points['square_vertices'] = pts

        # Draw a triangle
        if self.triangles != []:
            self.triangles = np.array(self.triangles)
            ind = np.argsort(self.triangles[:,1])
            self.triangles = self.triangles[ind,:]

            for i, segments in enumerate(self.triangles):
                pts = segments[0]
                score = segments[1]
                err = segments[2]
                color_mono = int(score*255)
                cv2.polylines(img_chk[3], [pts], True, color_mono, 1)

                if i == 0:
                    center = np.mean(pts,axis = 0, dtype = np.int32)
                    center = (center[0],center[1])
                    cv2.polylines(img, [pts], True, color, thickness)
                    cv2.putText(img,'triangle'+str(center),center, font, font_size ,color)
                    points['triangle_center'] = center
                    points['triangle_vertices'] = pts

            segments = np.array(self.triangles)[:,0]
            center = [np.mean(pts, axis = 0, dtype = np.int32) for pts in segments]
            #res = self.binning2D(center)
            #self.[:,:,0] = self.[:,:,0] + np.array(res.statistic)
	'''
        if np.max(self.[:,:,0]) > 30:
            ind = np.argmax(self.[:,:,0])
            tracking_bin = np.unravel_index(ind, (self.bins))
            self.tracking_posi = np.array(tracking_bin) * self.binsize

        if self.tracking_posi != 0:
            self.tracking
	'''
        # Draw images for check
        img_check = np.concatenate(
        (cv2.hconcat([img_chk[0],img_chk[1],img_chk[2]]),
         cv2.hconcat([img_chk[3],img_chk[4],img_chk[5]])), axis=0)

        #return img, img_check, points
	return img, points

    def my_approxPoly(self,cnt,nmin):
        cnt = cnt.reshape(-1, 2)
        n = len(cnt)
        cos_list = [self.angle_cos( cnt[(i-1)%n], cnt[i], cnt[(i+1)%n] ) for i in range(n)]
        while n > nmin:
            ind = np.argmin(cos_list)
            cnt = np.delete(cnt,ind,axis=0)
            cos_list = np.delete(cos_list,ind,axis=0)
            n = len(cnt)
            cos_list[(ind)%n] = self.angle_cos( cnt[(ind-1)%n], cnt[(ind)%n], cnt[(ind+1)%n] )
            cos_list[(ind-1)%n] = self.angle_cos( cnt[(ind-2)%n], cnt[(ind-1)%n], cnt[(ind)%n] )
        return cnt, cos_list

#    def hough_tracking(self, img):

    def shape_recognition(self, img):
        '''
        Recognition for circle, square, and triangle
        '''
        self.check = [[] for i in range(3)]
        self.squares = []
        self.triangles = []
        self.circles = []

        img = cv2.GaussianBlur(img, (5, 5), 0)
        im_area = self.im_x * self.im_y

        # RGB to Gray
        for gray in cv2.split(img):
            #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.check[0] = gray.copy()

            for thrs in self.thrs_list:

                # Create Binary Image
                if thrs == 0:
                    bin = cv2.Canny(gray, 0, 100)
                else:
                    _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                bin = cv2.dilate(bin, None)
                # Draw contours
                _img_dummy, contours, _hierarchy \
                    = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # For each contour, ...
                for cnt in contours:
                    self.check[1].append(cnt)
                    area = cv2.contourArea(cnt)
                    x,y,w,h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h

                    # Object filter based on area and aspect ratio
                    if area > self.area_min*im_area and \
                       area < self.area_max*im_area and \
                       aspect_ratio > self.aspect_ratio_min and \
                       aspect_ratio < self.aspect_ratio_max:
                        # Shape recognition
                        cnt_approx = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)
                        self.find_squares(cnt_approx)
                        self.find_triangles(cnt_approx)
                        self.find_circles(cnt_approx)
                        self.check[2].append(cnt)


	'''
        if self.squares != []:
            segments = np.array(self.squares)[:,0]
            center = [np.mean(pts, axis = 0, dtype = np.int32) for pts in segments]
            res = self.binning2D(center)
            self.[:,:,1] = self.[:,:,1] + np.array(res.statistic)

        if self.circles != []:
            segments = np.array(self.circles)[:,0]
            center = [pts for pts in segments]
            res = self.binning2D(center)
            self.[:,:,2] = self.[:,:,2] + np.array(res.statistic)
	'''

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
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, 15.0, (640,480))

    # Create video windows
    cv2.namedWindow(WINDOW_NAME)

    # ObjectDetection instance
    od = ObjectDetection(org_frame)

    while end_flag == True:
        # Object Detection
        res_frame, chk_frame, points = od.shape_recognition(org_frame)
        # Display the frame
        cv2.imshow(WINDOW_NAME, res_frame)
        cv2.imshow('For check', chk_frame)
        # Write the frame
        #out.write(res_frame)
        # Press Esc to stop
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break
        # Read the next frame
        end_flag, org_frame = cap.read()

    # When everything done, close windows and release the capture
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
