import numpy as np
import cv2

#load the required XML classifiers
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')

#load the image or the video
img = cv2.imread('./images/obama.png')
#img = cv2.imread('./images/obama2.png')
#img = cv2.imread('./images/selfie2.jpg')
#img = cv2.imread('./images/smallFace.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h).
# Once we get these locations, we can create a ROI for the face and apply eye and smile detection on this ROI.
faces = face_cascade.detectMultiScale(gray, 1.2, 2)
for (x,y,w,h) in faces:
    #defining thre ROI (face)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_face_gray = gray[y:y+h, x:x+w]
    roi_half_face_gray = gray[int(y+h/2):y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #detecting eyes
    eyes = eye_cascade.detectMultiScale(roi_face_gray, 1.15, 2)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #detecting smile, consider only below half of face ROI
    smile = smile_cascade.detectMultiScale(roi_half_face_gray,1.2, 2)
    for (sx,sy,sw,sh) in smile:
        cv2.rectangle(roi_color,(sx,sy+int(h/2)),(sx+sw,sy+int(h/2)+sh),(0,0,255),2)
cv2.imshow('img',img)
cv2.waitKey(10000)
cv2.destroyAllWindows()