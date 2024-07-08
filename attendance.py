import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
classes = []
img_list = os.listdir(path)
print(img_list)

for cl in img_list:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    classes.append(os.path.splitext(cl)[0])
print(classes)

def find_encodings(imgs):
    encodelist = []
    for image in imgs:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(image)
        if encodes: 
            encodelist.append(encodes[0])
    return encodelist

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDatalist=f.readlines()
        namelist=[]
        for line in myDatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        
        if name not in namelist:
            now=datetime.now()
            dt=now.strftime('%H:%M:%S: %D')
            f.writelines(f'\n{name},{dt}')



know_encode = find_encodings(images)
print('Encoding done')
print(len(know_encode))
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_s = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    face_Curr = face_recognition.face_locations(img_s)
    encode_curr = face_recognition.face_encodings(img_s, face_Curr)

    for encodedFace, faceloc in zip(encode_curr, face_Curr):
        matches = face_recognition.compare_faces(know_encode, encodedFace)
        face_dis = face_recognition.face_distance(know_encode, encodedFace)

        # print(face_dis)
        matchInd=np.argmin(face_dis)

        if matches[matchInd]:
            name=classes[matchInd].upper()
            # print(name)

            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('webcam',img)
    key=cv2.waitKey(10)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
