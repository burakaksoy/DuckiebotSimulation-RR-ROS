#!/usr/bin/env python
import rospy
import keyboard # pip3 install keyboard or pip install keyboard
from geometry_msgs.msg import Twist

class KeyTeleop(object):
    def __init__(self):
        self.pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        
        # Initialize ROS Node
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing......" %(self.node_name))
        
        # Setup parameters
        self.framerate = self.setupParam("~framerate",60.0)
        
        # Setup timer
        self.timer_img_low = rospy.Timer(rospy.Duration.from_sec(1.0/self.framerate),self.cbTimer)
        rospy.loginfo("[%s] Initialized." %(self.node_name))
        
    def setupParam(self,param_name,default_value):
        value = rospy.get_param(param_name,default_value)
        rospy.set_param(param_name,value) #Write to parameter server for transparancy
        rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value
        
    def cbTimer(self,event):
        if not rospy.is_shutdown():
            self.teleop_publish(self.pub_cmd)
            
    def teleop_publish(self,publisher):
        vel_cmd_msg = Twist()
        linear_vel = 0.0
        angular_vel = 0.0
        
        if keyboard.is_pressed('w'):
            linear_vel = 0.44
        if keyboard.is_pressed('s'):
            linear_vel = -0.44
        if keyboard.is_pressed('a'):
            angular_vel = 1.0
        if keyboard.is_pressed('d'):
            angular_vel = -1.0
        
        vel_cmd_msg.linear.x =  linear_vel
        vel_cmd_msg.linear.y =  0.0
        vel_cmd_msg.linear.z =  0.0
        vel_cmd_msg.angular.x =  0.0
        vel_cmd_msg.angular.y =  0.0
        vel_cmd_msg.angular.z =  angular_vel
        # Publish 
        publisher.publish(vel_cmd_msg)
            
            
    def onShutdown(self):
        rospy.loginfo("[%s] Shutdown." %(self.node_name))
        
    
if __name__ == '__main__': 
    rospy.init_node('duckiebot_sim_key_teleop',anonymous=False)
    key_teleop = KeyTeleop()
    rospy.on_shutdown(key_teleop.onShutdown)
    rospy.spin()
