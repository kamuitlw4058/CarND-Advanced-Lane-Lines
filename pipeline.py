# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: yang
"""

import os
import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
import line

def thresholding(img):
    #print(img.shape)
    #setting all sorts of thresholds
    #cv2.imshow("orig", img)
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=10 ,thresh_max=230)
    #cv2.imshow("x_thresh",x_thresh*255)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    #cv2.imshow("mag_thresh",mag_thresh*255)
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    #cv2.imshow("dir_thresh",dir_thresh*255)
    hls_thresh = utils.hls_select(img, thresh=(180, 255))
    #cv2.imshow("hls_thresh",hls_thresh*255)
    lab_thresh = utils.lab_select(img, thresh=(155, 200))
    #cv2.imshow("lab_thresh",lab_thresh*255)
    luv_thresh = utils.luv_select(img, thresh=(225, 255))
    #cv2.imshow("luv_thresh",luv_thresh)

    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1
    #cv2.imshow("threshholded", threshholded*255)
    #cv2.waitKey(0)
    return threshholded



def colorMask(name,img,hsv,lower,upper):


    # 创建掩膜
    mask = cv2.inRange(hsv, lower, upper)
    # 将原图像和掩膜做位与运算
    res = cv2.bitwise_and(img, img, mask=mask)

    #cv2.imshow(name, res)
    return res



def road_line(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 设置阈值下限和上限，去除背景颜色
    lower_red = np.array([80, 0, 0])
    upper_red = np.array([160, 255, 150])

    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])

    lower_yellow = np.array([100, 43, 46])
    upper_yellow = np.array([120, 255, 255])
    # 创建掩膜
    # 将原图像和掩膜做位与运算

    white_img = colorMask("white_mask", img, hsv, lower_white, upper_white)
    yellow_img = colorMask("yellow_mask", img, hsv, lower_yellow, upper_yellow)

    img = cv2.bitwise_or(white_img, yellow_img)

    #cv2.imshow("res", img)
    return img

def processing(img, object_points, img_points, M, Minv, left_line, right_line):
    # camera calibration, image distortion correction
    # undist = utils.cal_undistort(img,object_points,img_points)
    cv2.imshow("img", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # 设置阈值下限和上限，去除背景颜色
    lower_red = np.array([80, 0, 0])
    upper_red = np.array([160, 255, 150])


    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])


    lower_yellow = np.array([100,43,46])
    upper_yellow = np.array([120,255,255])

    lower_lane = np.array([0,0,50])
    upper_lane = np.array([180,43,120])
    # 创建掩膜
    # 将原图像和掩膜做位与运算

    white_img = colorMask("white_mask",img,hsv,lower_white,upper_white)
    # yellow_img = colorMask("yellow_mask", img, hsv, lower_yellow, upper_yellow)
    # lane_img = colorMask("lane_mask", img, hsv, lower_lane, upper_lane)


    undist = img
    gray = cv2.cvtColor(white_img,cv2.COLOR_RGB2GRAY)

    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    ret, binary2 = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    #cv2.imshow("binary", binary)
    rgb =cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    cv2.imshow("rgb", rgb)


    #cv2.imshow("binary2", binary)
    # get the thresholded binary image
    thresholded = thresholding(rgb)
    cv2.imshow("thresholded*255", thresholded * 255)
    #perform perspective  transform
    thresholded_wraped = cv2.warpPerspective(binary2, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    cv2.imshow("thresholded_wraped*255", thresholded_wraped*255)


    #perform detection
    if left_line.detected and right_line.detected:
        print("find line")
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        print("no find line")
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)

    #draw the detected laneline and the information
    area_img = utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    cv2.imshow("area_img", area_img)
    cv2.waitKey(50)

    #print(area_img)
    #curvature,pos_from_center = utils.calculate_curv_and_pos(thresholded_wraped,left_fit, right_fit)
    #result = utils.draw_values(area_img,curvature,pos_from_center)
    #result=area_img
    result = rgb
    return result
#
#
left_line = line.Line()
right_line = line.Line()
cal_imgs = utils.get_images_by_dir('camera_cal')
object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
M,Minv = utils.get_M_Minv()

project_outpath = 'vedio_out/test.mp4'
project_video_clip = VideoFileClip("test.avi")
project_video_out_clip = project_video_clip.fl_image(lambda clip: processing(clip,object_points,img_points,M,Minv,left_line,right_line))

project_video_out_clip.write_videofile(project_outpath, audio=False)


##draw the processed test image
# test_imgs = utils.get_images_by_dir('test_images')
# undistorted = []
# for img in test_imgs:
#    img = utils.cal_undistort(img,object_points,img_points)
#    undistorted.append(img)
#
# result=[]
# for img in undistorted:
#    res = processing(img,object_points,img_points,M,Minv,left_line,right_line)
#    result.append(res)
#
# plt.figure(figsize=(20,68))
# for i in range(len(result)):
#
#    plt.subplot(len(result),1,i+1)
#    plt.title('thresholded_wraped image')
#    plt.imshow(result[i][:,:,::-1])
    
    