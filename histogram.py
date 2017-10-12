import cv2
from chess_pieces.extra_tools import hog
from matplotlib import pyplot as plt
import numpy as np

white_img = cv2.imread('sliding_windows/sliding_window0.jpg')
black_img = cv2.imread('sliding_windows/sliding_window51.jpg')

# white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
# black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)

# mask = np.zeros(black_img.shape[:2], np.uint8)
# print np.shape(black_img)
# mask[np.shape(black_img)[0]/3:np.shape(black_img)[0], 0:np.shape(black_img)[1]] = 255
# masked_img = cv2.bitwise_and(black_img,black_img,mask = mask)
#
# white_hist = cv2.calcHist([white_img],[0],None,[256],[50,256])
# black_hist = cv2.calcHist([black_img],[0],None,[128],[0,128])
#
# # plt.plot(223)
# # plt.imshow(masked_img, 'gray')
#
# plt.figure()
# plt.title("white_hist")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(white_hist)
# plt.xlim([0, 256])
#
# plt.figure()
# plt.title("black_hist")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(black_hist)
# plt.xlim([0, 128])
#
# plt.show()


result = hog(black_img)
print len(result)
print type(result)
# cv2.imshow('s',result)
# cv2.waitKey(0)

plt.figure()
plt.plot(result)

plt.show()
