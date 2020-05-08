#!/usr/bin/env python
# This is a Gym Duckiebot Simulation Robot Raconteur Service in Python  

import time
import RobotRaconteur as RR
#Convenience shorthand to the default node.
#RRN is equivalent to RR.RobotRaconteurNode.s
RRN=RR.RobotRaconteurNode.s

import numpy as np
import cv2

import threading
import traceback
import sys
import argparse

import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
# from gym_duckietown.wrappers import UndistortWrapper

class gymDuckiebotSimRRService(object):
    def __init__(self):
        # Initialize Robot Simulation
        # Other Maps: udem1, straight_road, small_loop, loop_empty, 4way, zigzag_dists, loop_obstacles
        # loop_pedestrians, loop_dyn_duckiebots, regress_4way_adam
        self.env = DuckietownEnv( seed = 1, max_steps = 5000, map_name = 'zigzag_dists', draw_curve = False, draw_bbox = False, distortion = True )
        self.env.reset()
        self.env.render()
        self.action = np.array([0.0, 0.0])
        self.framerate = self.env.unwrapped.frame_rate
        
        # Initialize the camera images and control streaming
        self._lock=threading.RLock()
        self._streaming=False
        
        self._framestream=None
        self._framestream_endpoints=dict()
        self._framestream_endpoints_lock=threading.RLock()
        
        
    #Capture a frame, apply the action and return a CamImage structure to the client
    def CaptureFrame_n_Action(self):
        with self._lock:
            image=RRN.NewStructure("experimental.gymDuckiebotSim.CamImage")
            # Grab image from simulation and apply the action
            obs, reward, done, info = self.env.step(self.action)
            
            #if done:
            #    self.env.reset()                
            
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR) # Correct color for cv2
            # frame = obs

            image.width=frame.shape[1]
            image.height=frame.shape[0]
            image.step=frame.shape[1]*3
            image.data=frame.reshape(frame.size, order='C')

            return image
            
    #Start the thread that captures images and sends them through connected
    #FrameStream pipes
    def StartStreaming(self):
        if (self._streaming):
            raise Exception("Already streaming")
        self._streaming=True
        t=threading.Thread(target=self.frame_threadfunc)
        t.start()

    #Stop the streaming thread
    def StopStreaming(self):
        if (not self._streaming):
            raise Exception("Not streaming")
        self._streaming=False

    #FrameStream pipe member property getter and setter
    @property
    def FrameStream(self):
        return self._framestream
    @FrameStream.setter
    def FrameStream(self,value):
        self._framestream=value
        #Create the PipeBroadcaster and set backlog to 3 so packets
        #will be dropped if the transport is overloaded
        self._framestream_broadcaster=RR.PipeBroadcaster(value,3)
        
    #Function that will send a frame at ideally (self.framerate) fps, although in reality it
    #will be lower because Python is quite slow.  This is for
    #demonstration only...
    def frame_threadfunc(self):
        #Loop as long as we are streaming
        while(self._streaming):
            #Capture a frame
            try:
                frame = self.CaptureFrame_n_Action()
            except:
                #TODO: notify the client that streaming has failed
                self._streaming=False
                return
            #Send the new frame to the broadcaster.  Use AsyncSendPacket
            #and a blank handler.  We really don't care when the send finishes
            #since we are using the "backlog" flow control in the broadcaster.
            self._framestream_broadcaster.AsyncSendPacket(frame,lambda: None)

            # Put in a 100 ms delay
            time.sleep(1.0/self.framerate)     
                
    def setAction(self, v, w):
        with self._lock:
            # v = Forward Velocity [-1 1]
            # w = Steering angle [-1 1]
            self.action = np.array([v, w]) 
        
    def Shutdown(self):
        print("Duckiebot Simulation RR Service Shutdown.")
        self._streaming=False
        self.env.close()
        
        
def main():
    #Accept the names of the webcams and the nodename from command line            
    parser = argparse.ArgumentParser(description="Gym Duckiebot Simulation Robot Raconteur Service")
    parser.add_argument("--nodename",type=str,default="experimental.gymDuckiebotSim.DuckiebotSim",help="The NodeName to use")
    parser.add_argument("--tcp-port",type=int,default=2356,help="The listen TCP port") #random port, any unused port is fine
    parser.add_argument("--wait-signal",action='store_const',const=True,default=False)
    args = parser.parse_args()
    
    #Initialize the objects in the service
    obj = gymDuckiebotSimRRService()

    with RR.ServerNodeSetup(args.nodename,args.tcp_port) as node_setup:

        RRN.RegisterServiceTypeFromFile("experimental.gymDuckiebotSim") # This is the .robdef file
        RRN.RegisterService("DuckiebotSim","experimental.gymDuckiebotSim.DuckiebotSim",obj)
    
        # These are for using the service on Web Browsers
        node_setup.tcp_transport.AddWebSocketAllowedOrigin("http://localhost")
        node_setup.tcp_transport.AddWebSocketAllowedOrigin("http://localhost:8000")
        node_setup.tcp_transport.AddWebSocketAllowedOrigin("https://johnwason.github.io")
        
        if args.wait_signal:  
            #Wait for shutdown signal if running in service mode          
            print("Press Ctrl-C to quit...")
            import signal
            signal.sigwait([signal.SIGTERM,signal.SIGINT])
        else:
            #Wait for the user to shutdown the service
            if (sys.version_info > (3, 0)):
                input("Server started, press enter to quit...")
            else:
                raw_input("Server started, press enter to quit...")
    
        #Shutdown
        obj.Shutdown()

if __name__ == '__main__':
    main()
