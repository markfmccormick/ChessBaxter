#!/usr/bin/env python

import rospy
import baxter_interface
import time

rospy.init_node('Hello_Baxter')

right = baxter_interface.Limb('right')
right.set_joint_position_speed(0.8)
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

base_right = {'right_s0': 0.08, 'right_s1': -1.0,
	'right_e0': 1.19, 'right_e1': 1.94,
	'right_w0': -0.67, 'right_w1': 1.03, 'right_w2': 0.50}
base_left = {'left_s0': 1.0, 'left_s1': -1.0,
	'left_e0': -1.19, 'left_e1': 1.94,
	'left_w0': 0.67, 'left_w1': 1.03, 'left_w2': -0.50}


pivot_from_on = {'right_s0': 0.7, 'right_s1': -0.29,
	'right_e0': 0.0, 'right_e1': 1.2,
	'right_w0': 0.0, 'right_w1': 0.0, 'right_w2': 0.0}

pivot_from_above = {'right_s0': 0.7, 'right_s1': -0.3,
	'right_e0': 0.0, 'right_e1': 0.7,
	'right_w0': 0.0, 'right_w1': 0.5, 'right_w2': 0.0}

pivot_to_above = {'right_s0': 0.7, 'right_s1': -0.13,
	'right_e0': 0.0, 'right_e1': 0.0,
	'right_w0': 0.0, 'right_w1': 1.5, 'right_w2': 0.0}

pivot_to_on = {'right_s0': 0.7, 'right_s1': 0.165,
	'right_e0': 0.0, 'right_e1': 0.0,
	'right_w0': 0.0, 'right_w1': 1.2, 'right_w2': 0.0}

# s0 for angle,
# s1 for height
# e1 for elbow reach
# w1 for wrist height/reach
above = right.joint_angles()
above['right_s0']=0.7
above['right_s1']=0.15
above['right_e0']=0.0
above['right_e1']=0.0
above['right_w0']=0.0
above['right_w1']=1.2
above['right_w2']=0.0

on = right.joint_angles()
on['right_s0']=1.22
on['right_s1']=0.4
on['right_e0']=0.0
on['right_e1']=0.0
on['right_w0']=0.0
on['right_w1']=0.0
on['right_w2']=0.0

right.move_to_joint_positions(base_right)
left.move_to_joint_positions(base_left)

right_gripper = baxter_interface.Gripper('right')
# right_gripper.reboot()
right_gripper.calibrate()
# Percentage of maximum, default 50,40,30,5
right_gripper.set_parameters({"velocity":50.0, 
							"moving_force":40.0, 
							"holding_force":30.0,
							"dead_zone":5.0})
# right_gripper.open()
# right_gripper.close()

# right.move_to_joint_positions(above)
# right_gripper.open()
# right.move_to_joint_positions(on)
# right_gripper.close()
# time.sleep(0.25)
# right.move_to_joint_positions(above)

# right.move_to_joint_positions(on)
# right_gripper.open()
# time.sleep(0.25)
# right.move_to_joint_positions(above)


position_map = {}
right_joint_labels = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
with open("square_positions.txt") as position_labels:
	for line in position_labels:
		square_positions = {}
		joint_positions1 = {}
		joint_positions2 = {}
		if len(line) > 18:
			line = line.strip('\n')
			line = line.split(":")
			square = line[0]
			positions = line[1].split(";")
			values = positions[1].split(",")
			for i in range(len(right_joint_labels)):
				joint_positions1[right_joint_labels[i]] = float(values[i])
			square_positions[positions[0]] = joint_positions1
			values = positions[3].split(",")
			for i in range(len(right_joint_labels)):
				joint_positions2[right_joint_labels[i]] = float(values[i])
			square_positions[positions[2]] = joint_positions2
			position_map[square] = square_positions

# right.move_to_joint_positions(pivot_from_above)
# right.move_to_joint_positions(pivot_from_on)

pivot_points = ["a8", "a7", ]

#s0,s1,e0,e1,w0,w1,w2
#position = raw_input("Enter a square: ")
position = "b8"
print position_map[position]["above"]
print position_map[position]["on"]
right.move_to_joint_positions(position_map[position]["above"])
right_gripper.open()
right.move_to_joint_positions(position_map[position]["on"])
right_gripper.close()
time.sleep(0.25)
right.move_to_joint_positions(position_map[position]["above"])

right.move_to_joint_positions(pivot_from_above)
right.move_to_joint_positions(pivot_from_on)
right_gripper.open()
right.move_to_joint_positions(pivot_from_above)
right.move_to_joint_positions(pivot_to_above)
right.move_to_joint_positions(pivot_to_on)
right_gripper.close()
time.sleep(0.25)
right.move_to_joint_positions(pivot_to_above)

position = "g3"
right.move_to_joint_positions(position_map[position]["above"])
right.move_to_joint_positions(position_map[position]["on"])
right_gripper.open()
time.sleep(0.25)
right.move_to_joint_positions(position_map[position]["above"])

right.move_to_joint_positions(base_right)
left.move_to_joint_positions(base_left)
