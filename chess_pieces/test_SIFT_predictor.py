import numpy as np
import cv2
from SIFT_predictor import predict

img = cv2.imread("chess_pieces/white_king8.png")

a, max = predict(img)
print "\nPiece: " + str(a)
print "Max: " + str(max)

cv2.imshow("img", img)
cv2.waitKey(0)
