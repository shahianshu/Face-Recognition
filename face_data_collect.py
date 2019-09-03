import cv2 as cv
import numpy as np 


cap=cv.VideoCapture(0)

faceCascade=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_data=[]
skip=0
dataSet_path='./data/'
file_name=input('Enter your name:')

while True:

	ret,frame=cap.read()
	
	if ret==False:
		continue

	faces=faceCascade.detectMultiScale(frame,1.3,5)
	#print(faces)
	faces=sorted(faces,key=lambda f:f[2]*f[3])
	face_section=frame
	for face in faces[-1:]:
		x,y,w,h=face
		cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0,),2)


		offset=10 #padding on each side
		face_section=frame[y-offset:y+offset+h,x-offset:x+offset+w]
		face_section=cv.resize(face_section,(200,200))

		skip+=1
		if skip%10==0 :
			face_data.append(face_section)
			print(len(face_data))

	cv.imshow("frame",frame)
	cv.imshow("face section",face_section)
	key_pressed=cv.waitKey(1)&0xFF
	if key_pressed==ord('q'):
		break


face_data=np.asarray(face_data)
print(face_data.shape)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(dataSet_path+file_name+'.npy',face_data)


cap.release()
cv.destroyAllWindows()
