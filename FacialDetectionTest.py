#3450 Term Project by David Kenyon

#import statements
import cv2
import numpy as np

#haarcascades - Found via opencv github: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
catface_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

#Enable primary web camera (will be 0 for most laptops with an integrated webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    #Inititalize to detect in grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    catfaces = catface_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Face Detection
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #Eye detection
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #If program has gotten this far, display 'Human' text
            cv2.putText(img, 'Human', (0,80), font, 3, (200,255,255), 2, cv2.LINE_AA)
    #Cat Face Detection
    for (cx,cy,cw,ch) in catfaces:
        cv2.rectangle(img, (cx,cy), (cx+cw, cy+ch), (0,0,255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #If program has gotten this far, display 'Cat' text
        cv2.putText(img, 'Feline', (0,170), font, 3, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('img', img)
    #End program by hitting 'q'
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
