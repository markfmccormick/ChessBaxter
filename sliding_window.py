import cv2

def sliding_window(image, stepSize, windowSize):
    # slide window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])