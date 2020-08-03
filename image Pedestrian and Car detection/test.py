import cv2


#load image
img= cv2.imread('image.jpg')
car_classifier='car_classifier.xml'

#load pre-trained data from open cv using haar cascade algorithm
trained_car= cv2.CascadeClassifier(car_classifier)
trained_pedestrian= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")


grayscale_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pedestrian_grayscale=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_coordinates=trained_car.detectMultiScale(grayscale_img)
pedestrian_coordinates=trained_pedestrian.detectMultiScale(grayscale_img)
for(x,y,w,h) in car_coordinates:
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)
for(x,y,w,h) in pedestrian_coordinates:
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)

print(car_coordinates)
#show image
cv2.imshow('car detector',img)
cv2.waitKey()

print("code completed")


