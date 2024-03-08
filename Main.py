import cv2  # Step 1
import numpy as np  # Step 1
import face_recognition  # Step 1

imgVijay = face_recognition.load_image_file('ImagesBasic/vijay.jpg')   # Step 1
imgVijay = cv2.cvtColor(imgVijay, cv2.COLOR_BGR2RGB)  # Step 1
imgTest = face_recognition.load_image_file('ImagesBasic/abdul.jpg')   # Step 1
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)  # Step 1


faceLoc = face_recognition.face_locations(imgVijay)[0]  # Step 2
encodeVijay = face_recognition.face_encodings(imgVijay)[0]  # Step 2
cv2.rectangle(imgVijay,(faceLoc[3], faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)  # Step 2

print(faceLoc)  # Step 2


faceLocTest = face_recognition.face_locations(imgTest)[0]  # Step 2
encodeVijayTest = face_recognition.face_encodings(imgTest)[0]  # Step 2
cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)  # Step 2

results = face_recognition.compare_faces([encodeVijay],encodeVijayTest)  # Step 3
faceDis = face_recognition.face_distance([encodeVijay],encodeVijayTest)  # Step 4
print(results)  # Step 3
print(faceDis)  # Step 4

cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)  # Step 5

cv2.imshow('Vijay',imgVijay)  # Step 1
cv2.imshow('vijay test',imgTest)  # Step 1
cv2.waitKey(0)  # Step 1
