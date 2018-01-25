import cv2
import glob

for filename in glob.glob('training_data/videos/*.mp4'):

    video = cv2.VideoCapture(filename)
    dest = filename[21:-4]
    dest = "training_data/piece_data/" + dest

    success,image = video.read()
    count = 0
    # success = True
    while success and count < 3000:
        success,image = video.read()
        if count %5 == 0:
            print "Saving frame " + str(count) + " to "+dest+'\n'		
            cv2.imwrite(dest+"/frame"+str(count)+".jpg", image[:,450:-450])
        count +=1
