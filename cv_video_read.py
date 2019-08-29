import cv2 as cv

cap=cv.VideoCapture(0)
faceCascade=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
	ret,frame=cap.read()
	grayFrame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
	if ret==False :
		continue

	faces=faceCascade.detectMultiScale(frame,1.3,5)
	for (x,y,w,h) in faces :
		cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)

	cv.imshow('video frame',frame)
	key_pressed=cv.waitKey(1)&0xFF
	if key_pressed==ord('q') :
		break

cap.release()
cv.destroyAllWindows()
