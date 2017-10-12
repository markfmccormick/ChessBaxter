import numpy as np
import cv2

from extra_tools import extract_HOG, chess_train
from scipy.stats import itemfreq

# 'queen','knight','bishop','rook',
clf_PS = chess_train('descriptors/HOG', ['pawn', 'queen'], train_mod="SVM")
clf_KS = chess_train('descriptors/HOG', ['king', 'square'], train_mod="SVM")
clf_QS = chess_train('descriptors/HOG', ['queen', 'square'], train_mod="SVM")
clf_kS = chess_train('descriptors/HOG', ['knight', 'square'], train_mod="SVM")
clf_BS = chess_train('descriptors/HOG', ['bishop', 'square'], train_mod="SVM")
clf_RS = chess_train('descriptors/HOG', ['rook', 'square'], train_mod="SVM")
print "Training done!"

img = cv2.imread("pawn.png")
gray = img.copy() #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print "Compute descriptors"
kp, des1 = extract_HOG(gray, 0)
np.savetxt('test2.txt', des1, delimiter=',')
des1 = np.float32(des1)


label = ["Pawn", "King", "Queen", "Knight", "Bishop", "Rook", "Square"]
models = [clf_PS,clf_KS,clf_QS,clf_kS,clf_BS,clf_RS]

print ""
print "************************"
c_val = []
c_prob = []
for i, clf in enumerate(models):
    prediction = clf.predict(des1)
    freq = itemfreq(prediction)
    c_val.append(freq[:, 1].tolist())
    print freq
    pred_prob = clf.predict_proba(des1)
    # print pred_prob
    print "************************"
    out = pred_prob.mean(axis=0)
    print "Out: " + str(out)
    c_prob.append(out.tolist())
    print "************************"
    print "************************"
    print ""

# NOT WORKING: NEED TO FIX FREQUENCY ACC TO INCLUDE ALL CLASSES PER CLASSIFIER
# freq_class = np.array(c_val).flatten()
# prob_class = np.array(c_prob).flatten()
# prod = freq_class * prob_class
# idx_c = freq_class.argmax()
# print "Found: " + label[idx_c]
