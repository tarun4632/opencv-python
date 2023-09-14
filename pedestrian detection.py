import cv2
import numpy as np

video_capture = cv2.VideoCapture(r"Resources/Test_Video.mp4") #importing the video in the file

background_subtractor = cv2.createBackgroundSubtractorMOG2() #creating an object for background substraction

object_tracks = [] #creating an empty list to maintain object centres

while True:
    ret, img = video_capture.read() #reading a frame in the video

    move_obj = background_subtractor.apply(img) #applying the above substractor

    move_obj = cv2.erode(move_obj, None, iterations=1)
    move_obj = cv2.dilate(move_obj, None, iterations=1)
    imgcontour = img.copy()

    contours, _ = cv2.findContours(move_obj.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100: #eliminating the detection of small objects
            x, y, w, h = cv2.boundingRect(contour)
            object_center = (x + w // 2, y + h // 2)

            object_tracks.append(object_center)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img, object_center, 4, (0, 255, 0), -1)
        else:
            continue


    cv2.namedWindow("final", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("final",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN) #making the screen fullscreen
    key = cv2.waitKey(1)
    cv2.imshow('Pedestrian Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

