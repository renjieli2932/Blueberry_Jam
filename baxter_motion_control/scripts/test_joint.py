#!/usr/bin/env python

import argparse
import math
import random
import numpy as np
import rospy
import tf

from std_msgs.msg import (
    UInt16,
)

import baxter_interface

from baxter_interface import CHECK_VERSION

from ik_service_client import ik_test

class JointPositionController(object):

    def __init__(self):

        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)
        self._left_arm = baxter_interface.limb.Limb("left")
        self._right_arm = baxter_interface.limb.Limb("right")
        self._left_joint_names = self._left_arm.joint_names()
        self._right_joint_names = self._right_arm.joint_names()

        # control parameters
        self._rate = 500.0  # Hz

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        # set joint state publishing to 500Hz
        self._pub_rate.publish(self._rate)

    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._left_arm.exit_control_mode()
            self._right_arm.exit_control_mode()
            self._pub_rate.publish(100)  # 100Hz default joint state rate
            rate.sleep()
        return True

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self._left_arm.move_to_neutral()
        self._right_arm.move_to_neutral()

    def clean_shutdown(self):
        print("\nExiting example...")
        #return to normal
        self._reset_control_modes()
        self.set_neutral()
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

    def inverse_kinematics(self, cartesian_space):
        """
        Implement inverse_kinematics using "ik_service_client"
        """
        # Convert "cartesian-space" to "joint_space"
        # Data Type: dictionary {joint_name:joint_angle}
        joint_space = ik_test('left', cartesian_space)
        return [joint_space[joint] for joint in self._left_joint_names]

    def move(self,goal_position):
        """
        Main module for joint position control"
        """
        #Set a goal position as a joint space vector 'q'
        self.q_goal = self.inverse_kinematics(goal_position)

        rate = rospy.Rate(self._rate)
        start = rospy.Time.now()

        # Function: make desired positions along a straight line
        # between q_current and q_goal. The length the robot moves
        # in each time step is "eps"
        def make_pos(eps):
            q_current = [self._left_arm.joint_angle(joint)
                                        for joint in self._left_joint_names]
            delta = (np.array(self.q_goal) - np.array(q_current))
            delta_norm = np.linalg.norm(delta)
            if delta_norm > eps:
                delta = delta * eps / delta_norm
            return q_current + delta, delta_norm

        # Function: make a command
        # Data Type: dictionary {Joint Name:Joint Position}
        def make_cmd(joint_names):
            pos_list, err = make_pos(0.05)
            return dict([(joint, pos_list[i])
                    for i, joint in enumerate(joint_names)]), err

        # Move the robot
        print("Moving.")
        while not rospy.is_shutdown():
            self._pub_rate.publish(self._rate)
            cmd, err = make_cmd(self._left_joint_names)
            self._left_arm.set_joint_positions(cmd)
            rate.sleep()


def main():

    print("Initializing node... ")
    rospy.init_node("joint_position_controller")

    # Set a goal position
    # (This should be replaced with data from the kinect in the future)
    q = tf.transformations.quaternion_from_euler(0.0, math.pi, 0.0, axes='sxyz')
    goal_position = {'x':0.7,
                     'y':0.0,
                     'z':0.0,
                     'rotx':q[0],
                     'roty':q[1],
                     'rotz':q[2],
                     'rotw':q[3]}

    jpc = JointPositionController()
    rospy.on_shutdown(jpc.clean_shutdown)
    jpc.set_neutral()
    jpc.move(goal_position)

    print("Done.")

if __name__ == '__main__':
    main()
