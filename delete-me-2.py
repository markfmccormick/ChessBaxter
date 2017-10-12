from sklearn.externals import joblib
import numpy as np
import cv2

X = joblib.load('descriptors/ORB/pawn/pawn.pkl')

img = cv2.imread("white_pawn27.png")
orb = cv2.ORB_create(edgeThreshold=4)
kp1, des1 = orb.detectAndCompute(img, None)

out = clf.predict_proba(des1).mean(axis=0)
print "Out: " + str(out)
