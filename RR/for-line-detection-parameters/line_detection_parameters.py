#!/usr/bin/env python

# This script is for finding a good line detection in duckiebot simulation images

import numpy as np
import cv2
from collections import namedtuple
#########################################################################
# FUNCTIONS FOR LINE DETECTION: BEGIN
# Detect Lines and Return detected line segments
def detect_lines(frame, lines_img = False):
    #-------------------PARAMETERS: BEGIN
    image_size =  np.array([480,640])
    top_cutoff = 160   

    # image_size =  np.array([120,160])
    # top_cutoff = 40 

    canny_thresholds =  np.array([50,150]) #[50,150] [80,200]

    hsv_white1 =   np.array([80,10,140]) # [0,0,50] [0,0,150]
    hsv_white2 =   np.array([175,115,250]) # [180,120,255] [180,60,255])
    
    hsv_yellow1 =  np.array([0,40,150]) # [25,210,50]  [25,140,100]
    hsv_yellow2 =  np.array([55,230,245]) 
    
    # hsv_red1 =  np.array([0,140,100])
    # hsv_red2 =  np.array([15,255,255])
    hsv_red3 =  np.array([165,140,100])
    hsv_red4 =  np.array([180,255,255])
    
    
    dilation_kernel_size = 15 # 3
    # sobel_threshold = 40
    hough_threshold = 20
    hough_min_line_length = 3 # 3
    hough_max_line_gap = 1
    Detections = namedtuple('Detections', 'lines normals area centers')
    #-------------------PARAMETERS: END
    # Helper functions: BEGIN--------------------------
#    def _scaleandshift2(img, scale, shift):
#        img_shift = np.zeros(img.shape, dtype='float32')
#        for i in range(3):
#            s = np.array(scale[i]).astype('float32')
#            p = np.array(shift[i]).astype('float32')
#            np.multiply(img[:,:,i], s, out=img_shift[:, :, i])
#            img_shift[:, :, i] += p
#
#        return img_shift    
    
    def detectLines(color):
        bw, edge_color = _colorFilter(color)
        # lines, normals, centers = _lineFilter(bw, edge_color)
        lines = _HoughLine(edge_color)
        centers, normals = _findNormal(bw, lines)
        return Detections(lines=lines, normals=normals, area=bw, centers=centers)

    def _colorFilter(color):
        # binary dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_kernel_size, dilation_kernel_size))

        # threshold colors in HSV space
        if color == 'white':
            bw = cv2.inRange(hsv, hsv_white1, hsv_white2)
            cv2.imshow("WHITE Filter Image", bw)    
            # refine edge for certain color
            edge_color = cv2.bitwise_and(cv2.dilate(bw, kernel), edges)
            cv2.imshow("Filtered WHITE Edge", edge_color)

        elif color == 'yellow':
            bw = cv2.inRange(hsv, hsv_yellow1, hsv_yellow2)
            cv2.imshow("YELLOW Filter Image", bw)

            # refine edge for certain color
            edge_color = cv2.bitwise_and(cv2.dilate(bw, kernel), edges)
            cv2.imshow("Filtered YELLOW Edge", edge_color)
        elif color == 'red':
            # bw1 = cv2.inRange(hsv, hsv_red1, hsv_red2)
            bw2 = cv2.inRange(hsv, hsv_red3, hsv_red4)
            # bw = cv2.bitwise_or(bw1, bw2)
            bw = bw2
            cv2.imshow("RED Filter Image", bw)

            # refine edge for certain color
            edge_color = cv2.bitwise_and(cv2.dilate(bw, kernel), edges)
            cv2.imshow("Filtered RED Edge", edge_color)
        else:
            raise Exception('Error: Undefined color strings...')

        return bw, edge_color
    
    def _HoughLine(edge):
        lines = cv2.HoughLinesP(edge, 1, np.pi/180, hough_threshold, np.empty(1), hough_min_line_length, hough_max_line_gap)
        # lines = cv2.HoughLinesP(edge, 1, np.pi/180, hough_threshold, hough_min_line_length, hough_max_line_gap)
        if lines is not None:
            # print("ZAAAAAA")
            # print(lines)
            # lines = np.array(lines[0])
            lines = np.array(lines)
            lines = np.reshape(lines, (lines.shape[0],lines.shape[2]))
            print("lines.shape")
            print(lines.shape)
            # print("lines")
            # print(lines)

        else:
            lines = []
        return lines
    
    def _findNormal(bw, lines):
        normals = []
        centers = []
        if len(lines)>0:
            differ = lines[:, 0:2] -lines[:, 2:4]
            length = np.sqrt(np.sum(np.multiply(differ,differ), axis=1, keepdims=True))
            # print("length.shape")
            # print(length.shape)
            dx = 1.* (lines[:,3:4]-lines[:,1:2])/length
            dy = 1.* (lines[:,0:1]-lines[:,2:3])/length

            centers = np.hstack([(lines[:,0:1]+lines[:,2:3])/2, (lines[:,1:2]+lines[:,3:4])/2])
            x3 = (centers[:,0:1] - 3.*dx).astype('int')
            y3 = (centers[:,1:2] - 3.*dy).astype('int')
            x4 = (centers[:,0:1] + 3.*dx).astype('int')
            y4 = (centers[:,1:2] + 3.*dy).astype('int')
            x3 = _checkBounds(x3, bw.shape[1])
            y3 = _checkBounds(y3, bw.shape[0])
            x4 = _checkBounds(x4, bw.shape[1])
            y4 = _checkBounds(y4, bw.shape[0])
            flag_signs = (np.logical_and(bw[y3,x3]>0, bw[y4,x4]==0)).astype('int')*2-1
            normals = np.hstack([dx, dy]) * flag_signs
 
            lines = _correctPixelOrdering(lines, normals)
            # print("length.shape22")
            # print(lines.shape)
        return centers, normals
        
    def _checkBounds(val, bound):
        val[val<0]=0
        val[val>=bound]=bound-1
        return val
        
    def _correctPixelOrdering(lines, normals):
        lines2 = np.copy(lines)
        flag = ((lines[:,2]-lines[:,0])*normals[:,1] - (lines[:,3]-lines[:,1])*normals[:,0])>0
        for i in range(len(lines)):
            if flag[i]:
                x1,y1,x2,y2 = lines[i, :]
                lines2[i, :] = [x2,y2,x1,y1]
        return lines2
    
        

    def drawLines(bgr,lines, paint):
        if len(lines)>0:
            for x1,y1,x2,y2 in lines:
                cv2.line(bgr, (x1,y1), (x2,y2), paint, 5)
                cv2.circle(bgr, (x1,y1), 5, (0,255,0))
                cv2.circle(bgr, (x2,y2), 5, (0,0,255))
    # Helper functions: END--------------------------    
    # Resize image
    hei_original, wid_original = frame.shape[0:2]

    if image_size[0] != hei_original or image_size[1] != wid_original:
            frame = cv2.resize(frame, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Crop image
    frame = frame[top_cutoff:,:,:]
    
    #Blur Image
    #frame = cv2.blur(frame,(3,3))
    frame = cv2.GaussianBlur(frame,(7,7),0)
    # cv2.imshow("Blurred Image", frame)
    
    # Set image
    bgr = np.copy(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_to_show = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB) # To read the correct hsv values from rgb values
    # cv2.imshow("HSV image", hsv)
    cv2.imshow("HSV image to show", hsv_to_show)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GRAY image", gray)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # Find edges
    edges = cv2.Canny(gray, canny_thresholds[0], canny_thresholds[1], apertureSize = 3)
    cv2.imshow("Canny Edges", edges)
    
    # Detect lines and normals (as Detections namedtuple.)
    white = detectLines('white')
    yellow = detectLines('yellow')
    red = detectLines('red')
    
    if (lines_img):
        # View image with lines for debug purposes(add image_with_lines to returned objects)
        # Draw lines and normals
        image_with_lines = np.copy(frame)
        drawLines(image_with_lines, white.lines, (0, 0, 0))
        drawLines(image_with_lines, yellow.lines, (255, 0, 0))
        drawLines(image_with_lines, red.lines, (0, 255, 0))

    # Convert to normalized pixel coordinates, and add segments to segmentList
    arr_cutoff = np.array((0, top_cutoff, 0, top_cutoff))
    arr_ratio = np.array((1./image_size[1], 1./image_size[0], 1./image_size[1], 1./image_size[0]))

    if len(white.lines) > 0:
        lines_normalized_white = ((white.lines + arr_cutoff) * arr_ratio)
    else:
        lines_normalized_white = white.lines
    if len(yellow.lines) > 0:
        lines_normalized_yellow = ((yellow.lines + arr_cutoff) * arr_ratio)
    else:
        lines_normalized_yellow = yellow.lines
    if len(red.lines) > 0:
        lines_normalized_red = ((red.lines + arr_cutoff) * arr_ratio)
    else:
        lines_normalized_red = red.lines
    
    if (lines_img):
        return lines_normalized_white, white.normals, lines_normalized_yellow, yellow.normals, lines_normalized_red, red.normals,image_with_lines
    else:
        return lines_normalized_white, white.normals, lines_normalized_yellow, yellow.normals, lines_normalized_red, red.normals,frame
# FUNCTIONS FOR LINE DETECTION: END
########################################################################

def main():
    img_num = "016"
    print(img_num)
    frame = cv2.imread("./screen"+img_num+".png")

    is_view = True
    
    if is_view:
        cv2.namedWindow("Original Image")
        cv2.namedWindow("Image-with-lines") # Debug

    # Detect Lines and Return detected normed line segments and normals
    lns_white, nrmls_white, lns_yellow, nrmls_yellow, lns_red, nrmls_red, image = detect_lines(frame, lines_img = is_view)

    if is_view:
        cv2.imshow("Original Image", frame)
        cv2.imshow("Image-with-lines", image)

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image


if __name__ == '__main__':
    main()

