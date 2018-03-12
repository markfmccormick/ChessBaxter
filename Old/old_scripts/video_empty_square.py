import cv2

video = cv2.VideoCapture("training_data/videos/empty_square.mp4")

dest = "training_data/piece_data/empty_square/"
success, image = video.read()
count = 0
while success and count < 3000:
    if count % 2 == 0:
        success,image = video.read()
        cv2.imwrite(dest+"frame"+str(count)+".jpg", image)
    count+=1