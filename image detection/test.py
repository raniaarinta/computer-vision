import cv2


#load pre-trained data from open cv using haar cascade algorithm
trained_face= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#load image
img= cv2.imread('bts.jpg')

#convert grayscale
grayscale_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect face
face_coordinates=trained_face.data.detect(grayscale_img)

print(face_coordinates)
#show image
cv2.imshow('face detector', grayscale_img)
cv2.waitKey()

print("code completed")