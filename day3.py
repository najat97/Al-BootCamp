import numpy as np
import cv2
import matplotlib.pyplot as plt

image=cv2.imread('image3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray'),plt.show()

def convertToRGB(image):
  return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

face_coo = face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
print('Faces found :',len(face_coo))

for(x_face,y_face,w_face,h_face) in face_coo:
  cv2.rectangle(image,(x_face,y_face),(x_face+w_face,y_face+w_face),(0,0,255),10)

plt.imshow(convertToRGB(image)),plt.show()