import numpy as np
import cv2

from extra_tools import extract_HOG, chess_train
from scipy.stats import itemfreq

# 'queen','knight','bishop','rook',
clf_PKS = chess_train('descriptors/HOG', ['pawn', 'king', 'queen', 'knight','bishop', 'rook', 'square'], train_mod="SVM")
# clf_PKS = chess_train('descriptors/HOG', ['pawn', 'king', 'square'], train_mod="LR")
# clf_QkS = chess_train('descriptors/HOG', ['queen', 'knight', 'square'], train_mod="SVM")
# clf_BRS = chess_train('descriptors/HOG', ['bishop', 'rook', 'square'], train_mod="SVM")
print "Training done!"

img = cv2.imread("pawn.png")
gray = img.copy() #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print "Compute descriptors"
kp, des1 = extract_HOG(gray, 0)

np.savetxt('test2.txt', des1, delimiter=',')

des1 = np.float32(des1)

# cv2.imshow('img', img)
# cv2.waitKey(0)

# label = ["Pawn", "King", "Square", "Queen", "Knight", "Square", "Bishop", "Rook", "Square"]
label = ["Pawn", "King", "Queen", "Knight", "Bishop", "Rook", "Square"]
models = [clf_PKS,]# clf_QkS, clf_BRS]

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

# values = []
# counter = 0

# for entry in pred_prob:
#     if float(prediction[counter]) == 0.0:
#         values.append(entry[0])
#     counter += 1

# Calculate the mean
# print "Mean: " + str(sum(values)/len(values))

# out = pred_prob.mean(axis=0)
# print "Out: " + str(out)

# total = 0
# counter = 0
# for entry in clf.predict_proba(des1):
# 	if prediction[counter] == 1:
# 		total = total + entry[0]
# 	else:
# 		total = total + entry[1]
# 	counter += 1
#
# print "Mean2: " + str(total/counter)

# joblib.dump(clf,'classifiers/SIFT/dense/densepawnVking_classifier.pkl')
