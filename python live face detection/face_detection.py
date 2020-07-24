import cv2


#load pre-trained data from open cv using haar cascade algorithm
trained_face= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


#load webcam
webcam=cv2.VideoCapture(0)
while True:
    #read frame
    succesful_frame_read,frame= webcam.read()
    #convert_grayscale
    grayscale_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect 
    face_coordinates=trained_face.detectMultiScale(grayscale_img)
    #draw detected area
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
    cv2.imshow('face detector',frame)
    cv2.waitKey(1)

