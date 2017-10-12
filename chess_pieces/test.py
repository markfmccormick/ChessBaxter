import cv2

# Initiate ORB detector
orb = cv2.ORB_create(edgeThreshold=4)

image = cv2.imread('white_square/black_square11.png')

#cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

kp1, des1 = orb.detectAndCompute(image,None)

print len(kp1)
