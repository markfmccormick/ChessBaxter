import rospy
import baxter_interface
import time

"""
    This is the code for Baxter to physically make the given chess move.
    Takes the inital and final square, the position map of squares,
    Baxters left and right arm, right gripper as input, as well as the pivot state
    and capture state of the move. These are special cases for moves where a 
    pivot is needed, or a piece is being captured
"""

# pivot can be 'None', 'From' or 'To'
# capture is true or false
def perform_move(initial, final, position_map, right, left, gripper, pivot, capture):

    base_right = {'right_s0': 0.08, 'right_s1': -1.0, 
	    'right_e0': 1.19, 'right_e1': 1.94, 
	    'right_w0': -0.67, 'right_w1': 1.03, 'right_w2': 0.50}
    base_left = {'left_s0': 1.0, 'left_s1': -1.0, 
	    'left_e0': -1.19, 'left_e1': 1.94, 
	    'left_w0': 0.67, 'left_w1': 1.03, 'left_w2': -0.50}

    print "Capture: ", capture

    gripper.open()
    if capture == True:
        move_piece(final, "capture", position_map, right, gripper)
        if pivot == "From":
            move_piece(initial, "pivot_from", position_map, right, gripper)
            move_piece("pivot_to", final, position_map, right, gripper)
        elif pivot == "To":
            move_piece(initial, "pivot_to", position_map, right, gripper)
            move_piece("pivot_from", final, position_map, right, gripper)
        else:
            move_piece(initial, final, position_map, right, gripper)
    else:
        if pivot == "From":
            move_piece(initial, "pivot_from", position_map, right, gripper)
            move_piece("pivot_to", final, position_map, right, gripper)
        elif pivot == "To":
            move_piece(initial, "pivot_to", position_map, right, gripper)
            move_piece("pivot_from", final, position_map, right, gripper)
        else:
            move_piece(initial, final, position_map, right, gripper)
    
    right.move_to_joint_positions(base_right)
    left.move_to_joint_positions(base_left)

def move_piece(pos_1, pos_2, position_map, right, gripper):

    right.move_to_joint_positions(position_map[pos_1]["above"])
    right.move_to_joint_positions(position_map[pos_1]["on"])
    gripper.close()
    time.sleep(0.25)
    right.move_to_joint_positions(position_map[pos_1]["above"])

    right.move_to_joint_positions(position_map[pos_2]["above"])
    right.move_to_joint_positions(position_map[pos_2]["on"])
    gripper.open()
    time.sleep(0.25)
    right.move_to_joint_positions(position_map[pos_2]["above"])
    
    

    