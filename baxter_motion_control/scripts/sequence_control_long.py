#!/usr/bin/env python

import rospy
import numpy as np
from baxter_object_detection.msg import Object
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64, Bool,Empty
import baxter_interface
from sensor_msgs.msg import Image

import time

import argparse
import struct
import sys
import copy
import tf
import math
import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface


class PickAndPlace():
    def __init__(self, shape, desired_position, limb, hover_distance = 0.15, verbose=False):
        self._limb_name = limb # string
        self._shape_name = shape # string
        self._desired_position = desired_position # list[x,y] (integer)
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.limb.Limb(limb)
        self._limb_joint_names = self._limb.joint_names()
        self.prev_time = 0.0
        self.control_effort_x = 0.0
        self.control_effort_y = 0.0
        self.object_position = Object()
        self.hover_position = Pose()
	self.diff_old = 0.0
	self.zposi = 0.0

        self.step = 0
        self._rate = 10 #10Hz
	self.display_pub= rospy.Publisher('/robot/xdisplay',Image,queue_size=60)
	self.pub_grasp_now = rospy.Publisher("pump_on",Empty,queue_size=1)
	self.pub_release_now = rospy.Publisher("pump_off",Empty,queue_size=1)
	self.sub = rospy.Subscriber('/detected_image', Image,self.republish,None,1)
	#msg_grasp_now = Bool(False)
        #self.pub_grasp_now.publish(msg_grasp_now)

        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Moving to neutral pose...")
        self._limb.move_to_neutral()


    def republish(self,msg):
        """
            Sends the camera image to baxter's display
        """             
        self.display_pub.publish(msg)
    def callback_camera(self,msg):
        self.object_pose2Dpixel = msg
    def callback_kinect(self,msg):
        self.object_pose3D = msg
    def callback_control_effort_x(self,msg):
        self.control_effort_x = msg.data
    def callback_control_effort_y(self,msg):
        self.control_effort_y = msg.data

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def move_to_hover(self):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        hover = copy.deepcopy(self.object_pose3D)
        # approach with a pose the hover-distance above the requested pose
        hover.position.z = hover.position.z + self._hover_distance
        joint_angles = self.ik_request(hover)
        self._guarded_move_to_joint_position(joint_angles)
        self.step = self.step + 1
        rospy.sleep(1.0)

    def move_to_hover_2(self):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        hover = copy.deepcopy(self.object_pose3D)
        # approach with a pose the hover-distance above the requested pose
        hover.position.z = hover.position.z + self._hover_distance
	hover.position.y = hover.position.y + 0.2
        joint_angles = self.ik_request(hover)
        self._guarded_move_to_joint_position(joint_angles)
        self.step = self.step + 1
        rospy.sleep(1.0)

    def change_layer(self, delta_z):
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z - delta_z
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        self._guarded_move_to_joint_position(joint_angles)
        self.step = self.step + 1

    def approach(self, adjust_value= [0.0,0.0,0.0]):
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x - 0.055 + adjust_value[0]
        ik_pose.position.y = current_pose['position'].y + 0.015 + adjust_value[1]
        ik_pose.position.z = current_pose['position'].z
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        self._guarded_move_to_joint_position(joint_angles)
	'''
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = self.object_pose3D.position.z + adjust_value
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        self._guarded_move_to_joint_position(joint_angles)
	'''
	# adjust angle (counter moment)
        current_pose = self._limb.endpoint_pose()
        quaternion = (current_pose['orientation'].x,
	        current_pose['orientation'].y,
	        current_pose['orientation'].z,
	        current_pose['orientation'].w)
	euler = tf.transformations.euler_from_quaternion(quaternion)
	theta = euler[1] - 0.0*math.pi/180.0
	q = tf.transformations.quaternion_from_euler(euler[0], theta, euler[2], axes='sxyz')
	current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
	ik_pose.position.z = self.object_pose3D.position.z + adjust_value[2]
        #ik_pose.position.z = current_pose['position'].z
        ik_pose.orientation.x = q[0]
        ik_pose.orientation.y = q[1]
        ik_pose.orientation.z = q[2]
        ik_pose.orientation.w = q[3]
        joint_angles = self.ik_request(ik_pose)
        self._guarded_move_to_joint_position(joint_angles)

        self.step = self.step + 1

    def adjust(self,gain):
        now_time = time.time()
        freq = 1.0/(now_time - self.prev_time)
        self.prev_time = now_time
        #print(freq)

        if self._shape_name == "circle":
            center = self.object_pose2Dpixel.circle_center
            r = self.object_pose2Dpixel.circle_radius
        elif self._shape_name == "square":
            center = self.object_pose2Dpixel.square_center
        elif self._shape_name == "triangle":
            center = self.object_pose2Dpixel.triangle_center
        else:
            print("ERROR: INVALID SHAPE")

        if center:
	    
	    current_pose = self._limb.endpoint_pose()
	    if current_pose['position'].z < 0.03:
	    	self.zposi = 1.0
	    else:
	    	self.zposi = 0.0
	    print(current_pose['position'].z)
            # Subscribe Image from camera
            rospy.Subscriber("/pixel_x/control_effort", Float64, pnp.callback_control_effort_x)
            rospy.Subscriber("/pixel_y/control_effort", Float64, pnp.callback_control_effort_y)
            print(self.control_effort_x*gain)
            print(self.control_effort_y*gain)
            
            ik_pose = Pose()
            ik_pose.position.x = current_pose['position'].x + self.control_effort_y*gain
            ik_pose.position.y = current_pose['position'].y + self.control_effort_x*gain
            ik_pose.position.z = current_pose['position'].z - self.zposi #self.object_pose3D.position.z
            ik_pose.orientation.x = current_pose['orientation'].x
            ik_pose.orientation.y = current_pose['orientation'].y
            ik_pose.orientation.z = current_pose['orientation'].z
            ik_pose.orientation.w = current_pose['orientation'].w
            joint_angles = self.ik_request(ik_pose)

            #self._guarded_move_to_joint_position(joint_angles)
            self._limb.set_joint_positions(joint_angles)
            #self._limb.set_joint_velocities(cmd)

            diff = np.array(center) - np.array(desired_position)
            print('status=[{0},{1}]'.format(center[0],center[1]))
            print('desired=[{0},{1}]'.format(desired_position[0],desired_position[1]))
            print('desired=[{0},{1}]'.format(diff[0],diff[1]))
            if np.linalg.norm(diff) <= 7:
		#count = count + 1
		#if
                self.step = self.step + 1
	    self.diff_old = diff
            pub_state = rospy.Publisher("/pixel_x/state",Float64,queue_size=1)
            msg_state = Float64(center[0])
            pub_state.publish(msg_state)
            pub_state = rospy.Publisher("/pixel_y/state",Float64,queue_size=1)
            msg_state = Float64(center[1])
            pub_state.publish(msg_state)

            pub_setpoint = rospy.Publisher("/pixel_x/setpoint",Float64,queue_size=1)
            msg_setpoint = Float64(desired_position[0])
            pub_setpoint.publish(msg_setpoint)
            pub_setpoint = rospy.Publisher("/pixel_y/setpoint",Float64,queue_size=1)
            msg_setpoint = Float64(desired_position[1])
            pub_setpoint.publish(msg_setpoint)

    def retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)
        self.step = self.step + 1

    def pause(self):
        print("Pause. Ctrl-c to quit")

    def grasp(self):
        #msg_grasp_now = Bool(True)
        #self.pub_grasp_now.publish(msg_grasp_now)
	self.pub_grasp_now.publish()
        rospy.sleep(10.0)
	# TEST
	#self.pub_release_now.publish()
	# TEST OVER
        self.step = self.step + 1

    def release(self):
        #msg_grasp_now = Bool(True)
        #self.pub_grasp_now.publish(msg_grasp_now)
	self.pub_release_now.publish()
	rospy.sleep(3.0)
	# TEST OVER
        self.step = self.step + 1
	
    def main(self):
        rate = rospy.Rate(self._rate)

        # Motion Start
        while not rospy.is_shutdown():
            if self.step == 0:	    
	    	print("\nHovering...")
	        self.move_to_hover()
	    
	    #elif self.step == 1:
	    #	 print("\nAdjusting...")
            #    self.adjust(0.0003)
            elif self.step == 1:
                print("\nChange Layer...")
                self.change_layer(0.0)
	    #elif self.step == 3:
	    #	print("\nAdjusting...")
            #    self.adjust(0.0003)
            #elif self.step == 4:
            #    print("\nChange Layer...")
            #    self.change_layer(0.0)
            #elif self.step == 5:
	    #    print("\nAdjusting...")
            #    self.adjust(0.0003)
            #elif self.step == 6:
            #    print("\nChange Layer...")
            #    self.change_layer(0.0)
            elif self.step == 2:
		print("\nAdjusting...")
                self.adjust(0.0003)
		                
	    elif self.step == 3:
                print("\nApproaching...")
                self.approach()
	    elif self.step == 4:
                print("\nGrasping...")
                self.grasp()
            elif self.step == 5:
                print("\nRetracting...")
                self.retract()
	    elif self.step == 6:
	    	print("\nMoving to the hole...")
	        self.move_to_hover_2()

            #elif self.step == 7:
            #    print("\nAdjusting...")
            #    self.adjust(0.0003)            
            elif self.step == 7:
                print("\nChange Layer...")
                self.change_layer(0.0)
	    elif self.step == 8:
		print("\nAdjusting...")
                self.adjust(0.0003)
            #elif self.step == 10:
            #    print("\nChange Layer...")
            #    self.change_layer(0.02)
            #elif self.step == 11:
	    #    print("\nAdjusting...")
            #    self.adjust(0.0003)
            #elif self.step == 12:
            #    print("\nChange Layer...")
            #    self.change_layer(0.02)
            #elif self.step == 13:
	    #    print("\nAdjusting...")
            #    self.adjust(0.0003)
	
            elif self.step == 9:
                print("\nApproaching...")
                self.approach([0.005,-0.005,0.05])
            elif self.step == 10:
                print("\nReleasing...")
                self.release()
            elif self.step == 11:
                print("\nRetracting...")
                self.retract()

            else:
                self.pause()
                break
            rate.sleep()

        return 0

'''
def load_gazebo_models(table_pose=Pose(position=Point(x=1.0, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))
'''
if __name__ == '__main__':
    rospy.init_node('feedback_control', anonymous=True)

    # Load Gazebo Models via Spawning Services
    # load_gazebo_models()
    # Remove models from the scene on shutdown
    # rospy.on_shutdown(delete_gazebo_models)
    # Wait for the All Clear from emulator startup
    # rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'left'
    shape = 'circle'
    hover_distance = 0.15
    desired_position = [320,250]
    pnp = PickAndPlace(shape,desired_position,limb,hover_distance)

    pub_pid_enable = rospy.Publisher("/pid_enable",Bool,queue_size=1)
    msg_pid_enable = Bool(True)
    pub_pid_enable.publish(msg_pid_enable)

    # Subscribe Image from camera
    rospy.Subscriber("/detected_object", Object, pnp.callback_camera)

    # Subscribe Image from camera
    # rospy.Subscriber("/control_effort", Float64, pnp.callback_control_effort)

    # Subscribe Image from kinect
    #rospy.Subscriber("/xxxxxxx", Pose, pnp.callback_kinect)
    quaternion = tf.transformations.quaternion_from_euler(0.0, math.pi, 0.0, axes='sxyz')
    hover = Pose()
    hover.position.x = 0.6
    hover.position.y = 0.1
    hover.position.z = -0.04
    hover.orientation.x = quaternion[0]
    hover.orientation.y = quaternion[1]
    hover.orientation.z = quaternion[2]
    hover.orientation.w = quaternion[3]
    pnp.callback_kinect(hover)

    pnp.main()

    rospy.spin()
