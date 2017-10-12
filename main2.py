import numpy as np
import cv2
import math
from chessboard_detector import chessboard_homography

img_with_mathes, img_with_homography, points = chessboard_homography()

#print points

corner1 = points[0]
corner2 = points[1]
corner3 = points[16]
corner4 = points[-1]

print "corner" + str(corner1)
print corner2
print corner3
print corner4

cv2.circle(img_with_homography, (corner1), 10, (255, 0, 0), 10)
cv2.circle(img_with_homography, (corner2), 10, (255, 0, 0), 10)
cv2.circle(img_with_homography, (corner3), 10, (255, 0, 0), 10)
cv2.circle(img_with_homography, (corner4), 10, (255, 0, 0), 10)

## Calculate the area of the trapezoid, which is more or less the shape
## of the chessboard in the image:
## A = (a+b)/2*h        being 'a' and 'b' the bases of the trapezoid and 'h' its height
## Very approximate as the height is simply one of the two sides
a = math.sqrt((corner1[1]-corner3[1])**2+(corner1[0]-corner3[0])**2)
print "a: " + str(a)
b = math.sqrt((corner2[1]-corner4[1])**2+(corner2[0]-corner4[0])**2)
print "b: " + str(b)
h = math.sqrt((corner1[1]-corner2[1])**2+(corner1[0]-corner2[0])**2)
print "h: " + str(h)
area = (a+b)/2*h
print "area: " + str(area)

single_square = area/64
single_square_side = math.sqrt(single_square)
print "one square measures: " + str(single_square)
cv2.line(img_with_homography,corner1,(int(corner1[0]+single_square_side),int(corner1[1]+single_square_side)),0,3)

c = 0
#while(c<len(actual_list_of_points)-1):
 #   pt1 = actual_list_of_points[c]
  #  pt2 = actual_list_of_points[c+1]
   # cv2.line(img2,pt1,pt2,0,3)
    #c += 2

#cv2.imshow('Chessboard Detection',img_with_homography)
cv2.namedWindow('Chessboard Detection',cv2.WINDOW_NORMAL)
cv2.imshow('Chessboard Detection',img_with_homography)
cv2.resizeWindow('Chessboard Detection', 1568,820)
cv2.waitKey(0)
cv2.destroyAllWindows()
