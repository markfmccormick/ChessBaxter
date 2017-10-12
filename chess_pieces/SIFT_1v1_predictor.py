import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib


def predict(img1):

    dictionary = {0:"king", 1:"pawn"}
    siftdesc = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = siftdesc.detectAndCompute(img1, None)
    # for kp in kp1:
        # cv2.circle(img1, (int(kp.pt[0]),int(kp.pt[1])), 1, (255, 0, 0), 1)

    # cv2.imshow("img1", img1)
    # cv2.waitKey(0)

    # If enough descriptors were found, run classifiers and choose the most confident
    if np.size(des1) > 50:
        c = 0


        ### Load classifiers from a pickle file ###
        classifiers = []
        classifiers.append(joblib.load('chess_pieces/classifiers/SIFT/kingVpawn_classifier.pkl'))
        classifiers.append(joblib.load('chess_pieces/classifiers/SIFT/pawnVking_classifier.pkl'))


        max = float(0.0)
        piece_counter = 0
        piece = ""
        for classifier in classifiers:
            confidence = classifier.predict_proba(des1)

            float_confidence = float(confidence[0][0])
            if float_confidence> max:
                max = float_confidence
                piece = piece_counter

            print dictionary[piece_counter] + str(confidence[0][0])
            piece_counter += 1

        # If a piece was detected, return the name of that piece and the classifier's confidence
        if max > 0.75:
            return dictionary[piece], max

    return "None", 0
