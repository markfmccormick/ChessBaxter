from chessboard_detector import chessboard_homography
import cv2

img_with_colours, img_with_matches, img_with_homography, points = chessboard_homography()

cv2.namedWindow('Chessboard Detection',cv2.WINDOW_NORMAL)
cv2.imshow('Chessboard Detection',img_with_homography)
cv2.resizeWindow('Chessboard Detection', 1568,820)
cv2.waitKey(0)
cv2.destroyAllWindows()
