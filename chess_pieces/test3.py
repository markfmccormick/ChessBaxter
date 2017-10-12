import numpy as np
import cv2
from sklearn.externals import joblib
from extra_tools import extract_HOG


# # Initiate SIFT detector
# siftdesc = cv2.xfeatures2d.SIFT_create()

# # Initiate HOG detector
# hog = cv2.HOGDescriptor()

# Initiate ORB detector
orb = cv2.ORB_create(edgeThreshold=4)

#############
# IMAGES AND DESCRIPTORS
unseen_pawn_img = cv2.imread('another_pawn.jpg')
# kp1, des1 = siftdesc.detectAndCompute(unseen_pawn_img, None)
kp1, des1 = orb.detectAndCompute(unseen_pawn_img, None)
des1 = np.array(des1)
print np.shape(des1)

seen_pawn_img = cv2.imread('cropped_pictures/pawn/pawn34.png')
# kp2, des2 = siftdesc.detectAndCompute(seen_pawn_img, None)
kp2, des2 = orb.detectAndCompute(seen_pawn_img, None)
des2 = np.array(des2)

unseen_king_img = cv2.imread('king.jpg')
# kp3, des3 = siftdesc.detectAndCompute(unseen_king_img, None)
kp3, des3 = orb.detectAndCompute(unseen_king_img, None)

seen_king_img = cv2.imread('cropped_pictures/king/king38.png')
# kp4, des4 = siftdesc.detectAndCompute(seen_king_img, None)
kp4, des4 = orb.detectAndCompute(seen_king_img, None)

#############

print "SVM"
clf_svm = joblib.load('classifiers_test/ORB/pawnVall_svm_classifier.pkl')
print 'unseen_pawn_img'
print clf_svm.predict(des1), "Out: ", clf_svm.predict_proba(des1).mean(axis=0)
print 'seen_pawn_img'
print clf_svm.predict(des2), "Out: ", clf_svm.predict_proba(des2).mean(axis=0)
print 'unseen_king_img'
print clf_svm.predict(des3), "Out: ", clf_svm.predict_proba(des3).mean(axis=0)
print 'seen_king_img'
print clf_svm.predict(des4), "Out: ", clf_svm.predict_proba(des4).mean(axis=0)



print "\n\nLR"
clf_lr = joblib.load('classifiers_test/ORB/pawnVall_lr_classifier.pkl')
print 'unseen_pawn_img'
print clf_lr.predict(des1), "Out: ", clf_lr.predict_proba(des1).mean(axis=0)
print 'seen_pawn_img'
print clf_lr.predict(des2), "Out: ", clf_lr.predict_proba(des2).mean(axis=0)
print 'unseen_king_img'
print clf_lr.predict(des3), "Out: ", clf_lr.predict_proba(des3).mean(axis=0)
print 'seen_king_img'
print clf_lr.predict(des4), "Out: ", clf_lr.predict_proba(des4).mean(axis=0)



print "\n\nKNN"
clf_knn = joblib.load('classifiers_test/ORB/pawnVall_knn_classifier.pkl')
print 'unseen_pawn_img'
print clf_knn.predict(des1), "Out: ", clf_knn.predict_proba(des1).mean(axis=0)
print 'seen_pawn_img'
print clf_knn.predict(des2), "Out: ", clf_knn.predict_proba(des2).mean(axis=0)
print 'unseen_king_img'
print clf_knn.predict(des3), "Out: ", clf_knn.predict_proba(des3).mean(axis=0)
print 'seen_king_img'
print clf_knn.predict(des4), "Out: ", clf_knn.predict_proba(des4).mean(axis=0)
