#!/usr/bin/env python

#Robot Raconteur Gym Duckiebot Simulation client
#This program will show a live streamed image from
#the Cam and allow the user to drive the robot with w a s d keyboard input, q exits 

from RobotRaconteur.Client import *

import time
import numpy as np
import cv2
import sys
import keyboard # pip3 install keyboard or pip install keyboard

from collections import namedtuple
import math
from math import floor, atan2, pi, cos, sin, sqrt

#Function to take the data structure returned from the Webcam service
#and convert it to an OpenCV array
def WebcamImageToMat(image):
    frame2=image.data.reshape([image.height, image.width, 3], order='C')
    # print([image.height, image.width])
    return frame2
##########################################################################
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

#########################################################################
# FUNCTIONS FOR GROUND PROJECTION: BEGIN
# Convert image points to Ground Frame Coordinate points
def ground_projection(lns_white, lns_yellow, lns_red):
    #-------------------PARAMETERS: BEGIN
    # Extrinsic paramaters
    homography =  np.array([-7.552577e-06, -0.0002441265, -0.2141686, 0.001033941, -2.328867e-05, -0.3232461, -0.000191772, -0.007839249, 1])
    # Intrinsic parameters
    camera_matrix = np.array([332.92166660353803, 0.0, 322.3328995392407, 0.0, 327.51948074131144, 208.77795212351865, 0.0, 0.0, 1.0])
    # Distortion Coefficients
    distortion_coefficients = np.array([-0.2602548460819323, 0.045338226035521415, -0.0014496731529041933, 0.0007231160055086537,0.0])
    # Projection parameters
    projection_matrix = np.array([225.0209503173828, 0.0, 321.38889831269444, 0.0, 0.0, 260.9706115722656, 189.29544335933315, 0.0, 0.0, 0.0, 1.0, 0.0])
    # Rectification parameters
    rectification_matrix = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    # Image size
    image_height = 480
    image_width = 640
    
    #-------------------PARAMETERS: END

    # Helper functions: BEGIN--------------------------
    def image2ground_list(lines):
        lines_ground = []
        if len(lines)>0:     
            for x1,y1,x2,y2 in lines:
                x1_ground, y1_ground = image2ground(x1,y1)
                x2_ground, y2_ground = image2ground(x2,y2)
                lines_ground.append((x1_ground, y1_ground, x2_ground, y2_ground))
            lines_ground = np.array(lines_ground)
        return lines_ground
    # Input: a normalized point x-y, Output: ground frame point x-y 
    def image2ground(x,y):
        # Convert the point into image pixels from normalized coordinates
        u = int(image_width*x)
        v = int(image_height*y)
        # Boundary check (might be redundant, but just in case)
        u = _checkBound(u, image_width)
        v = _checkBound(v, image_height)
        # Get rid of fisheye distortion
        pt = np.array([u,v], dtype=np.float32)
        pt = pt.reshape((1,1,2))
        xy = cv2.undistortPoints(pt,cameraMatrix,distCoeffs,None,R,P)     
        x = xy.item(0)
        y = xy.item(1)
        # Take homoraphy to find ground coordinate frame points
        xyz = H_.dot(np.array([x,y,1.]))
        x = xyz[0]/xyz[2]
        y = xyz[1]/xyz[2]
        return x, y
                
    def _checkBound(val, bound):
        if val < 0:
            val = 0
        if val >= bound:
            val = bound-1
        return val
    # Helper functions: END--------------------------

    # Create H_ matrix for homography
    H_ = homography.reshape((3, 3))
    # Create inverse(H_)
    Hinv_ = np.linalg.inv(H_)
    # Create Camera Matrix
    cameraMatrix = camera_matrix.reshape((3, 3))
    # Distortion Coefficients
    distCoeffs = distortion_coefficients
    # Projection Matrix
    P = projection_matrix.reshape((3, 4))
    # Rectification Matrix
    R = rectification_matrix.reshape((3, 3))
    
    lns_white_ground = image2ground_list(lns_white)
    lns_yellow_ground = image2ground_list(lns_yellow)
    lns_red_ground = image2ground_list(lns_red)

    return lns_white_ground,lns_yellow_ground,lns_red_ground
# FUNCTIONS FOR GROUND PROJECTION: END
########################################################################


#########################################################################
# FUNCTIONS FOR LANE FILTER: BEGIN
# Calculate phi and d from ground frame projected lines
def lane_filter(p_lns_white, p_lns_yellow):
    #-------------------PARAMETERS: BEGIN 
    d_max = 0.3 # DEFAULT
    d_min = -0.15 # DEFAULT
    delta_d = 0.02 # in meters
    phi_min = -1.5
    phi_max = 1.5 
    delta_phi = 0.05 # DEFAULT
    linewidth_white = 0.05
    linewidth_yellow = 0.025
    lanewidth = 0.20
    min_max = 0.3 
    min_max = 0.1 # DEFAULT
    min_segs = 10
    #-------------------PARAMETERS: END

    # Helper functions: BEGIN--------------------------
    def generateVote(x1,y1,x2,y2,color):
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        t_hat = (p2-p1)/np.linalg.norm(p2-p1)
        n_hat = np.array([-t_hat[1],t_hat[0]])
        d1 = np.inner(n_hat,p1)
        d2 = np.inner(n_hat,p2)
        l1 = np.inner(t_hat,p1)
        l2 = np.inner(t_hat,p2)
        if (l1 < 0):
            l1 = -l1;
        if (l2 < 0):
            l2 = -l2;
        l_i = (l1+l2)/2
        d_i = (d1+d2)/2
        phi_i = np.arcsin(t_hat[1])
        if color == 'white': # right lane is white
            if(p1[0] > p2[0]): # right edge of white lane
                d_i = d_i - linewidth_white
            else: # left edge of white lane
                d_i = - d_i
                phi_i = -phi_i
            d_i = d_i - lanewidth/2

        elif color == 'yellow': # left lane is yellow
            if (p2[0] > p1[0]): # left edge of yellow lane
                d_i = d_i - linewidth_yellow
                phi_i = -phi_i
            else: # right edge of white lane
                d_i = -d_i
            d_i =  lanewidth/2 - d_i

        return d_i, phi_i, l_i
    # Helper functions: END--------------------------
    d, phi = np.mgrid[d_min:d_max:delta_d,phi_min:phi_max:delta_phi]
    # initialize measurement likelihood
    measurement_likelihood = np.zeros(d.shape)
    # Eliminate line segments behind the vehicle (Check whether x's are less than 0)
    if len(p_lns_white)>0:
        p_lns_white = p_lns_white[ p_lns_white[:,0] >= 0 ]
    if len(p_lns_yellow)>0:    
        p_lns_yellow = p_lns_yellow[ p_lns_yellow[:,0] >= 0 ]
    
    # Take the votes of white lines
    if len(p_lns_white)>0:     
        for x1,y1,x2,y2 in p_lns_white:
            # Generate the vote for each line segment
            d_i,phi_i,l_i = generateVote(x1,y1,x2,y2,'white')
            
            if d_i > d_max or d_i < d_min or phi_i < phi_min or phi_i>phi_max:
                continue
                
            i = floor((d_i - d_min)/delta_d)
            j = floor((phi_i - phi_min)/delta_phi)
            
            # Update measurement likelihood
            measurement_likelihood[int(i),int(j)] = measurement_likelihood[int(i),int(j)] +  1/(l_i)
    # Take the votes of yellow lines
    if len(p_lns_yellow)>0:     
        for x1,y1,x2,y2 in p_lns_yellow:
            # Generate the vote for each line segment
            d_i,phi_i,l_i = generateVote(x1,y1,x2,y2,'yellow')
            
            if d_i > d_max or d_i < d_min or phi_i < phi_min or phi_i>phi_max:
                continue
                
            i = floor((d_i - d_min)/delta_d)
            j = floor((phi_i - phi_min)/delta_phi)
            
            # Update measurement likelihood
            measurement_likelihood[int(i),int(j)] = measurement_likelihood[int(i),int(j)] +  1/(l_i)
            
    if np.linalg.norm(measurement_likelihood) == 0:
            return None,None,None
    # Generate the belief
    beliefRV = measurement_likelihood/np.sum(measurement_likelihood)
        
    maxids = np.unravel_index(beliefRV.argmax(),beliefRV.shape)
    # Calculate final d and phi
    d = d_min + maxids[0]*delta_d
    phi = phi_min + maxids[1]*delta_phi
    
    max_val = beliefRV.max()
    # Boolean to check whether in lane or not
    in_lane = max_val > min_max and (len(p_lns_white)+len(p_lns_yellow)) > min_segs and np.linalg.norm(measurement_likelihood) != 0
    
    return d,phi,in_lane
# FUNCTIONS FOR LANE FILTER: END
########################################################################

#########################################################################
# FUNCTIONS FOR LANE CONTROLLER: BEGIN
# # Calculate w and v from d and phi
def lane_controller(d,phi):
    #-------------------PARAMETERS: BEGIN
    v_bar = 0.3864
    k_d = -10.30
    k_theta = -5.15
    theta_thres = 0.523
    d_thres =  0.2615
    d_offset = 0.0
    
    #v_bar = 0.5 # nominal speed, 0.5m/s
    #k_theta = -2.0
    #k_d = - (k_theta ** 2) / ( 4.0 * v_bar)
    #theta_thres = math.pi / 6
    #d_thres = math.fabs(k_theta / k_d) * theta_thres
    #d_offset = 0.0    
    #-------------------PARAMETERS: END

    # Helper functions: BEGIN--------------------------
    # Helper functions: END--------------------------
    cross_track_err = d - d_offset
    heading_err = phi
    
    v = v_bar/2.0
    
    if math.fabs(cross_track_err) > d_thres:
        cross_track_err = cross_track_err / math.fabs(cross_track_err) * d_thres
    
    omega =  (k_d * cross_track_err + k_theta * heading_err)/5.0 #1.25
    
    return v,omega
# FUNCTIONS FOR LANE CONTROLLER: END
########################################################################

##########################################################################
current_frame = None

def main():

    url='rr+tcp://localhost:2356?service=DuckiebotSim'
    if (len(sys.argv)>=2):
        url=sys.argv[1]

    #Startup, connect, and pull out the camera from the objref    
    c = RRN.ConnectService(url)

    #Connect the pipe FrameStream to get the PipeEndpoint p
    p = c.FrameStream.Connect(-1)

    #Set the callback for when a new pipe packet is received to the
    #new_frame function
    p.PacketReceivedEvent += new_frame
    try:
        c.StartStreaming()
    except: pass
    
    is_view = True
    
    if is_view:
        cv2.namedWindow("Duckiebot Sim Image with RR")
        cv2.namedWindow("Image-with-lines") # Debug
    
    linear_vel = 0.0
    angular_vel = 0.0

    while True:
        #Just loop resetting the frame
        #This is not ideal but good enough for demonstration

        if (not current_frame is None):
            cv2.imshow("Duckiebot Sim Image with RR", current_frame)
        if cv2.waitKey(1) == ord('q'):
            break
            
        # Use frame from now on to prevent unknown updates on current frame.
        frame = current_frame
        
        # Detect Lines and Return detected normed line segments and normals
        lns_white, nrmls_white, lns_yellow, nrmls_yellow, lns_red, nrmls_red, image = detect_lines(frame, lines_img = is_view)
        # print(lns_white, nrmls_white)
        
        # Convert image points to Ground Frame Coordinate points (+x:ahead, +y:left)
        p_lns_white, p_lns_yellow, p_lns_red  = ground_projection(lns_white, lns_yellow, lns_red)
        # print(p_lns_yellow)
        
        # Calculate phi and d from detected lines(+d: left of lane, -d: right of lane, +phi: towards yellow, -phi: towards white)
        d,phi,in_lane = lane_filter(p_lns_white, p_lns_yellow)
        #print(d,phi,in_lane)
        
        # If d or phi could not be find, stop the car.
        if d is None or phi is None or in_lane is None:
            # linear_vel = 0.0
            # angular_vel = 0.0

            # Manual control
            if keyboard.is_pressed('w'):
                linear_vel = 0.44
            if keyboard.is_pressed('s'):
                linear_vel = -0.44
            if keyboard.is_pressed('a'):
                angular_vel = 1.0
            if keyboard.is_pressed('d'):
                angular_vel = -1.0
            if keyboard.is_pressed('x'):
                linear_vel = 0.0
                angular_vel = 0.0
            if keyboard.is_pressed('c'):
                cv2.imwrite('screen.png', frame)
                #from PIL import Image
                #im = Image.fromarray(frame)
                #im.save('screen.png')
                
            c.setAction(linear_vel, angular_vel)
            continue
        
        # Calculate w(omega) and v from d and phi
        v,w = lane_controller(d,phi)
        print(v,w)
        
        linear_vel = v
        angular_vel = w
        
        # Manual control
        if keyboard.is_pressed('w'):
            linear_vel = 0.44
        if keyboard.is_pressed('s'):
            linear_vel = -0.44
        if keyboard.is_pressed('a'):
            angular_vel = 1.0
        if keyboard.is_pressed('d'):
            angular_vel = -1.0
        if keyboard.is_pressed('x'):
            linear_vel = 0.0
            angular_vel = 0.0
        if keyboard.is_pressed('c'):
            cv2.imwrite('screen.png', frame)
            #from PIL import Image
            #im = Image.fromarray(frame)
            #im.save('screen.png')
            
        c.setAction(linear_vel, angular_vel)
        
        # View for Debug
        if is_view:
            image = cv2.resize(image, (640, 320))
            cv2.imshow("Image-with-lines",image)
            if cv2.waitKey(1)== ord('q'):
                break
                        
    # End of while 
    cv2.destroyAllWindows()

    p.Close()
    c.StopStreaming()

#This function is called when a new pipe packet arrives
def new_frame(pipe_ep):
    global current_frame

    #Loop to get the newest frame
    while (pipe_ep.Available > 0):
        #Receive the packet
        image=pipe_ep.ReceivePacket()
        #Convert the packet to an image and set the global variable
        current_frame = WebcamImageToMat(image)

if __name__ == '__main__':
    main()
