#!/usr/bin/env python

import rospy
import baxter_interface

rospy.init_node('Hello_Baxter')

right = baxter_interface.Limb('right')
left = baxter_interface.Limb('left')

# above = right.joint_angles()
# above['right_s0']=0.0
# above['right_s1']=0.0
# above['right_e0']=0.0
# above['right_e1']=0.0
# above['right_w0']=0.0
# above['right_w1']=0.0
# above['right_w2']=0.0
# above1={}
# above1['left_s0']=0.0
# above1['left_s1']=0.0
# above1['left_e0']=0.0
# above1['left_e1']=0.0
# above1['left_w0']=0.0
# above1['left_w1']=0.0
# above1['left_w2']=0.0
# right.move_to_joint_positions(above)
# left.move_to_joint_positions(above1)

left_gripper = baxter_interface.Gripper('right')
left_gripper.reboot()
left_gripper.calibrate()
left_gripper.open()

base_right = {'right_s0': 0.08, 'right_s1': -1.0,
	'right_e0': 1.19, 'right_e1': 1.94,
	'right_w0': -0.67, 'right_w1': 1.03, 'right_w2': 0.50}
base_left = {'left_s0': 1.0, 'left_s1': -1.0,
	'left_e0': -1.19, 'left_e1': 1.94,
	'left_w0': 0.67, 'left_w1': 1.03, 'left_w2': -0.50}

above = right.joint_angles()
above['right_s0']=1.0
above['right_s1']=-0.7
above['right_e0']=0.0
above['right_e1']=1.5
above['right_w0']=0.0
above['right_w1']=0.5
above['right_w2']=0.0

on = right.joint_angles()
on['right_s0']=1.01
on['right_s1']=-0.5
on['right_e0']=0.0
on['right_e1']=0.0
on['right_w0']=0.0
on['right_w1']=1.1
on['right_w2']=0.0

# right.move_to_joint_positions(base_right)
# left.move_to_joint_positions(base_left)

#right.move_to_joint_positions(above)
# right.move_to_joint_positions(on)
# right.move_to_joint_positions(above)
