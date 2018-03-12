import cv2
import glob

for filename in glob.glob("training_data/videos/*.mp4"):

    video = cv2.VideoCapture(filename)
    dest = filename[21:-4]
    dest = "training_data/piece_data/"+dest+"/_board_"
    print dest

    success, image = video.read()
    count = 0

    while success and count < 2500:
        success, image = video.read()
        if count % 3 == 0:
            cv2.imwrite(dest+"frame"+str(count)+".jpg", image[:,450:-450])
        count+=1