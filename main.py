import cv2
import numpy as np
import face_recognition


image_modi=face_recognition.load_image_file('images/Modiji.jpg')
image_modi=cv2.cvtColor(image_modi,cv2.COLOR_BGR2RGB)

test_modi=face_recognition.load_image_file('images/test_modi.jpg')
test_modi=cv2.cvtColor(test_modi,cv2.COLOR_BGR2RGB)
# test_modi=face_recognition.load_image_file('images/amit.jpg')
# test_modi=cv2.cvtColor(test_modi,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(image_modi)[0]
encode_modi=face_recognition.face_encodings(image_modi)[0]
cv2.rectangle(image_modi,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)


faceloc_test=face_recognition.face_locations(test_modi)[0]
encode_modi_test=face_recognition.face_encodings(test_modi)[0]
cv2.rectangle(test_modi,(faceloc_test[3],faceloc_test[0]),(faceloc_test[1],faceloc_test[2]),(255,0,255),2)


result=face_recognition.compare_faces([encode_modi],encode_modi_test)
faceDis=face_recognition.face_distance([encode_modi],encode_modi_test)
print(result,faceDis)

cv2.putText(test_modi,f'{result}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('image_modi',image_modi)
cv2.imshow('test_modi',test_modi)
cv2.waitKey(0)