__author__ = "Gerardo Aragon-Camarasa (gerardo.aragoncamarasa@glasgow.ac.uk"  # Any issue or problem, e-mail me :)

import os, copy
import numpy as np
import cv2
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

class DropletDet(object):
    def __init__(self, img_classes, db_dir="simple_db/"):
        self.img_classes = img_classes
        self.db_dir = db_dir
        self.colorSpace = cv2.COLOR_RGB2HSV
        self.colorSpaceInv = cv2.COLOR_HSV2RGB
        self._debug_output = False # Saves am image with the dectected droplets (countours)
        self._img_resize = 1  # Factor used to resize images, 1=same resolution
        self.cnt_petry = None
        self.prev_droplets = None
        (major, minor, _) = cv2.__version__.split(".")
        self.opencv_version = int(major)

    def generate(self, mask_exclude="empty_space", scale=1.5, minSize=(30, 30)):
        '''

        Generates an image database with masks and (optional) creates an image pyramid with corresponding masks. Images
        are saved in db_dir (defined within the class)

        Args:
            mask_exclude: class name of which a mask is not needed
            scale: resize factor to construct an image pyramid (not a full scale-space)
            minSize: minimum image size for the top level of the image pyramid. If set bigger than the image size,
                     only the original image is used

        Returns:
            None

        '''

        source_path = self.db_dir
        for (dirpath, dirnames, filenames) in os.walk(source_path):
            for filename in filenames:
                if filename.endswith('.png'):
                    print('')
                    img_filename = os.path.join(dirpath, filename)
                    print('Processing {} ...'.format(img_filename))

                    img = cv2.imread(img_filename)
                    os.remove(img_filename)

                    # img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

                    if mask_exclude not in dirpath:
                        # img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
                        timg = cv2.cvtColor(img, self.colorSpace)
                        timg = cv2.medianBlur(timg, 5)
                        timg = cv2.cvtColor(timg, cv2.COLOR_RGB2GRAY)  # TODO: CHECK THIS
                        ret2, th = cv2.threshold(timg, np.min(timg), np.max(timg), cv2.THRESH_OTSU)
                        th[th > 0] = 255
                        th = self._clean_image(th)

                    # cv2.imshow("original", img)
                    # cv2.imshow("mask", th)
                    # cv2.waitKey(0)

                    base, ext = os.path.splitext(filename)
                    level = 0
                    img_path = dirpath + "/" + base + "_" + str(level) + ext
                    cv2.imwrite(img_path, img)
                    if mask_exclude not in dirpath:
                        mask_path = dirpath + "/" + base + "_" + str(level) + "_mask" + ext
                        cv2.imwrite(mask_path, th)

                    while True:
                        w = int(img.shape[0] / scale)
                        h = int(img.shape[1] / scale)
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

                        if img.shape[0] < minSize[1] or img.shape[1] < minSize[0]:
                            break

                        level += 1
                        img_path = dirpath + "/" + base + "_" + str(level) + ext
                        cv2.imwrite(img_path, img)
                        if mask_exclude not in dirpath:
                            th = cv2.resize(th, (w, h), interpolation=cv2.INTER_LINEAR)
                            mask_path = dirpath + "/" + base + "_" + str(level) + "_mask" + ext
                            cv2.imwrite(mask_path, th)
                            th = self._clean_image(th)

    def get_descriptors(self, out_dir_desc, size_desc=5, stepSize=1, winSize=5, words_per_img = -1, specific_class=None):
        '''

        Computes descriptors for each image in database. The folder database is declared in db_dir within the class

        Args:
            out_dir_desc: destination folder where descriptor files will be saved for each image in database
            size_desc: descriptor length such that final length is: size_desc*size_desc*3
            step_size: number of pixels set to sample the image, stepSize=1 means that all pixels are considered
            winSize: sampling window size in pixels
            words_per_img: number of words set to describe the image. This reduces and compresses visual information to
                           prototypical visual words.
                           words_per_img=-1 means that the number of words will equal to the 10% of the total number of
                           descriptors extracted
                           words_per_img=None means that all extracted descriptors are saved and no compression is performed
            specific_class: If needed, only descriptors for the denoted class will be extracred. Class name should be
                            the same as the folder name.

        Returns:
            None

        '''

        if not os.path.exists(out_dir_desc):
            os.makedirs(out_dir_desc)
        # else:
        #     shutil.rmtree(out_dir_desc)
        #     os.makedirs(out_dir_desc)

        dynamic_words = False
        compute_words = True
        if words_per_img is not None:
            if words_per_img < 0:
                dynamic_words = True
        else:
            compute_words = False

        if specific_class is not None:
            source_path = os.path.join(self.db_dir, specific_class)
        else:
            source_path = self.db_dir

        for (dirpath, dirnames, filenames) in os.walk(source_path):
            for filename in filenames:
                if filename.endswith('.png') and 'mask' not in filename:

                    print('')
                    img_filename = os.path.join(dirpath, filename)
                    print('Processing {} ...'.format(img_filename))

                    # select target class
                    target = -1
                    for i, query in enumerate(self.img_classes):
                        if query in dirpath:
                            target = i

                    base, ext = os.path.splitext(filename)
                    mask_path = dirpath + "/" + base + "_mask" + ext

                    img = cv2.imread(img_filename)
                    img = cv2.cvtColor(img, self.colorSpace)

                    if os.path.exists(mask_path):
                        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY)
                    else:
                        mask = np.ones((img.shape[0], img.shape[1], 1), np.uint8) * 255

                    descs = []
                    shiftScore = np.uint8(np.floor(winSize / 2))
                    for (y, x, window) in self._sliding_window(img, stepSize=stepSize, windowSize=(winSize, winSize)):
                        if y + shiftScore < window.shape[0] - winSize or x + shiftScore < window.shape[1] - winSize:
                            continue

                        if window.shape[0] != winSize or window.shape[1] != winSize:
                            continue

                        if mask[y + shiftScore, x + shiftScore] == 0:
                            continue

                        desc = self._getDescriptor(window, size_desc)
                        descs.append(desc)

                    descs = np.float32(descs)

                    try:
                        print("length of descriptors: " + str(len(descs[0])))
                        print("number of descriptors extracted: " + str(len(descs)))

                        if compute_words:
                            if dynamic_words:
                                words_per_img = int(np.round(len(descs) * 0.1))

                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                            try:
                                ret, label, centers = cv2.kmeans(np.array(descs), words_per_img, criteria, 10,
                                                                 cv2.KMEANS_RANDOM_CENTERS)
                            except:
                                ret, label, centers = cv2.kmeans(np.array(descs), words_per_img, None, criteria, 10,
                                                                 cv2.KMEANS_RANDOM_CENTERS)

                            print("From " + str(len(descs)) + " descriptors to " + str(len(centers)) + " descriptors")
                            descs = np.float32(centers)

                        np.savetxt(os.path.join(out_dir_desc, str(target) + "_" + base + ".desc"), descs, delimiter=',')
                        # np.savetxt(os.path.join(out_dir_desc, base + ".class"), targets, delimiter=',')
                    except:
                        print("Cannot compute descriptors for"+img_filename)

    def get_descriptors_nowindow(self, out_dir_desc, size_desc=5, stepSize=1, winSize=5, words_per_img=-1, specific_class=None):
        '''

        Computes descriptors for each image in database without subsampling the image. This is useful when images are
        relatively small. The folder database is declared in db_dir within the class

        Args:
            out_dir_desc: destination folder where descriptor files will be saved for each image in database
            size_desc: descriptor length such that final length is: size_desc*size_desc*3
            step_size: number of pixels set to sample the image, stepSize=1 means that all pixels are considered
            winSize: sampling window size in pixels
            words_per_img: number of words set to describe the image. This reduces and compresses visual information to
                           prototypical visual words.
                           words_per_img=-1 means that the number of words will equal to the 10% of the total number of
                           descriptors extracted
                           words_per_img=None means that all extracted descriptors are saved and no compression is performed
            specific_class: If needed, only descriptors for the denoted class will be extracred. Class name should be
                            the same as the folder name.

        Returns:
            None

        '''

        if not os.path.exists(out_dir_desc):
            os.makedirs(out_dir_desc)
        # else:
        #     shutil.rmtree(out_dir_desc)
        #     os.makedirs(out_dir_desc)

        dynamic_words = False
        compute_words = True
        if words_per_img is not None:
            if words_per_img < 0:
                dynamic_words = True
        else:
            compute_words = False

        if specific_class is not None:
            source_path = os.path.join(self.db_dir, specific_class)
        else:
            source_path = self.db_dir

        for (dirpath, dirnames, filenames) in os.walk(source_path):
            for filename in filenames:
                if filename.endswith('.png') and 'mask' not in filename:

                    print('')
                    img_filename = os.path.join(dirpath, filename)
                    print('Processing {} ...'.format(img_filename))

                    # select target class
                    target = -1
                    for i, query in enumerate(self.img_classes):
                        if query in dirpath:
                            target = i

                    base, ext = os.path.splitext(filename)
                    mask_path = dirpath + "/" + base + "_mask" + ext

                    img = cv2.imread(img_filename)
                    img = cv2.cvt...
