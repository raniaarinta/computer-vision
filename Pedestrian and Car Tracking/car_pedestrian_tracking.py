import cv2
car_classifier='car_classifier.xml'


#load pre-trained data from open cv using haar cascade algorithm

trained_car= cv2.CascadeClassifier(car_classifier)
trained_pedestrian= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

#load image
video= cv2.VideoCapture('video.mp4')


while True:
    #read frame
    succesful_frame_read,frame= video.read()
    #convert_grayscale
    grayscale_vid= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect 
    car_coordinates=trained_car.detectMultiScale(grayscale_vid)
    pedestrian_coor=trained_pedestrian.detectMultiScale(grayscale_vid)
    #draw detected area
    for(x,y,w,h) in car_coordinates:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
    for(x,y,w,h) in pedestrian_coor:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,255),2)
    
    cv2.imshow('face detector',frame)
    cv2.waitKey(1)

"""
#detect face
face_coordinates=trained_face.detectMultiScale(grayscale_img)

#draw the regtangle
for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)

print(face_coordinates)
#show image
cv2.imshow('face detector',img)
cv2.waitKey()

print("code completed")"""