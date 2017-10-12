from sklearn.externals import joblib
import numpy as np
import cv2

clf = joblib.load('classifiers/ORB/pawnVking_classifier.pkl')

img = cv2.imread("white_pawn27.png")
cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
orb = cv2.ORB_create(edgeThreshold=4)
kp1, des1 = orb.detectAndCompute(img, None)

print clf.predict(des1)
out = clf.predict_proba(des1).mean(axis=0)
print "Out: " + str(out)
