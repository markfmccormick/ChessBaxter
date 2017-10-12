import numpy as np
import cv2
from ORB_predictor import predict

img = cv2.imread("chess_pieces/tower.png")

a, max = predict(img)
print "\nPiece: " + str(a)
print "Max: " + str(max)

cv2.imshow("img", img)
cv2.waitKey(0)
