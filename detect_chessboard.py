import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
    Used to detect where in the given image path the chessboard is
    Returns two sets of keypoints, the corners and centers of each square,
    as pixel coordinates of their locations in the new image.
"""

# Displays a visualisation of where the chessboard was found in the image
def display_detection(dst, img1, img2, matchesMask, keypoints, square_keypoints, center_keypoints, kp1, kp2, good):
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    for point in keypoints[0]:
        cv2.circle(img1, (int(point[0]),int(point[1])), 3, (255,0,0), -1)
    for point in square_keypoints[0]:
        cv2.circle(img2, (int(point[0]),int(point[1])), 3, (255,0,0), -1)
    for point in center_keypoints[0]:
        cv2.circle(img2, (int(point[0]),int(point[1])), 3, (255,0,0), -1)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()

# Based on the code in this tutorial: https://docs.opencv.org/3.3.0/d1/de0/tutorial_py_feature_homography.html
# Returns sets of keypoints corresponding to the the center and corners of the chess squares
def get_keypoints(imgpath):

    show_detection = False

    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('chessboard.png',0)  #query images
    img2 = cv2.imread(imgpath, 0)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Use SIFT to find the keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # Check if enough matches were found to detect the chessboard
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # Increased ransacReprojThreshold from 5.0 to 10.0, improved results with test images
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    keypoints = [[]]
    with open('data/chessboard_keypoints.txt') as chessboard_keypoints:
        for line in chessboard_keypoints:
            line = line.strip('\n')
            line = line.split(',')
            keypoints[0].append([line[0], line[1]])

    keypoints = np.array(keypoints, dtype="float32")
    square_keypoints = cv2.perspectiveTransform(keypoints, M)

    keypoints = [[]]
    with open('data/chessboard_keypoints_center.txt') as chessboard_keypoints:
        for line in chessboard_keypoints:
            line = line.strip('\n')
            line = line.split(',')
            keypoints[0].append([line[0], line[1]])
    
    keypoints = np.array(keypoints, dtype="float32")
    center_keypoints = cv2.perspectiveTransform(keypoints, M)

    if show_detection:
        display_detection(dst, img1, img2, matchesMask, keypoints, square_keypoints, center_keypoints, kp1, kp2, good)

    return square_keypoints, center_keypoints
