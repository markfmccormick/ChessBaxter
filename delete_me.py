from classify_colour import black_or_white
import cv2
import numpy as np
from chess_pieces.extra_tools import hog

img = cv2.imread('sliding_windows/with_colours/sliding_window52.jpg')
# print np.shape(img)
# print type(img)
# cv2.imshow('img',img)
# cv2.waitKey(0)
result, proba = black_or_white(img)
print result, proba
