import cv2
import time
import imutils
from PIL import Image
from numpy import asarray
vs = cv2.VideoCapture(0)
time.sleep(1)

while True:
	_,img = vs.read()
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	faces=face_cascade.detectMultiScale(img,scaleFactor=1.10,minNeighbors=1)
	for x,y,w,h in faces:
		img=cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0),1)	
	cv2.imshow("Face Detector", img)
	key = cv2.waitKey(2) & 0xFF
	if key == ord("q"):
		break
vs.release()
cv2.destroyAllWindows()

