import numpy as np
import cv2
import pickle
import glob
from sklearn.externals import joblib
from extra_tools import extract_HOG

folders_names = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook', 'square']

words_per_class = 200

print "\nNumber of descriptors:\n"
# Go through each folder containing pictures of the chess pieces
for folder_name in folders_names:
    # Read all images in the current folder
    print folder_name
    images = []

    for filename in glob.glob('cropped_pictures/' + folder_name + '/*.png'):
        image = cv2.imread(filename)
        gray = image.copy() #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(gray)

    descriptors = []

    for image in images:
        # add hog
        kp, des1 = extract_HOG(image, 0)

        for c, des in enumerate(des1):
            descriptors.append(des)

    print folder_name + ": " + str(len(descriptors))
    # print descriptors
    descs = np.float32(descriptors)
    joblib.dump(descs, 'descriptors/HOG/' + folder_name + '/' + folder_name + '.pkl')

    print("length of descriptors: " + str(len(descs[0])))
    print("number of descriptors extracted: " + str(len(descs)))

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        ret, label, centers = cv2.kmeans(np.array(descs), words_per_class, criteria, 10,
                                         cv2.KMEANS_RANDOM_CENTERS)
    except:
        ret, label, centers = cv2.kmeans(np.array(descs), words_per_class, None, criteria, 10,
                                         cv2.KMEANS_RANDOM_CENTERS)

    print("From " + str(len(descs)) + " descriptors to " + str(len(centers)) + " descriptors \n")
    descs = np.float32(centers)

    np.savetxt('descriptors/HOG/' + folder_name + '/' + folder_name + '.vocab', descs, delimiter=',')
