import rospy
import baxter_interface
import time

def perform_move(initial_square, final_square, position_map):
    rospy.init_node('Perform_Move')
    right = baxter_interface.Limb('right')
    left = baxter_interface.Limb('left')
    right.set_joint_position_speed(0.8)
    left.set_joint_position_speed(0.8)
    right_gripper = baxter_interface.Gripper('right')
    right_gripper.calibrate()
    right_gripper.set_parameters({"velocity":50.0, 
							"moving_force":20.0, 
							"holding_force":10.0,
							"dead_zone":5.0})

    base_right = {'right_s0': 0.08, 'right_s1': -1.0, 
	    'right_e0': 1.19, 'right_e1': 1.94, 
	    'right_w0': -0.67, 'right_w1': 1.03, 'right_w2': 0.50}
    base_left = {'left_s0': 1.0, 'left_s1': -1.0, 
	    'left_e0': -1.19, 'left_e1': 1.94, 
	    'left_w0': 0.67, 'left_w1': 1.03, 'left_w2': -0.50}

    right.move_to_joint_positions(base_right)
    left.move_to_joint_positions(base_left)

    right.move_to_joint_positions(position_map[initial_square]["above"])
    right_gripper.open()
    right.move_to_joint_positions(position_map[initial_square]["on"])
    right_gripper.close()
    time.sleep(0.25)   
    right.move_to_joint_positions(position_map[initial_square]["above"])

    right.move_to_joint_positions(position_map[final_square]["above"])
    right.move_to_joint_positions(position_map[final_square]["on"])
    right_gripper.open()
    time.sleep(0.25)
    right.move_to_joint_positions(position_map[final_square]["above"])

    right.move_to_joint_positions(base_right)
    left.move_to_joint_positions(base_left)
    

    