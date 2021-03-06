import cv2
car_classifier='car_classifier.xml'


#load pre-trained data from open cv using haar cascade algorithm

trained_car= cv2.CascadeClassifier(car_classifier)
trained_pedestrian= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

#load image
video= cv2.VideoCapture('TeslaAutopilotDashcamCompilation2018Versionmp4.mp4')


while True:
    #read frame
    succesful_frame_read,frame= video.read()
    #convert_grayscale
    grayscale_vid= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect 
    car_coordinates=trained_car.detectMultiScale(grayscale_vid)
    #draw detected area
    for(x,y,w,h) in car_coordinates:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
  
    
    cv2.imshow('face detector',frame)
    cv2.waitKey(1)


