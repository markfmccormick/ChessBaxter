import cv2

video = cv2.VideoCapture('training_data/test/test.mp4')

success,image = video.read()
count = 0
success = True
while success:
    success,image = video.read()
    print("Read a new frame: ", success)
    cv2.imwrite("training_data/test/frame%d.jpg" % count, image)
    count +=1
