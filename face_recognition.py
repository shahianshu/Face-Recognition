import cv2 as cv
import os
import numpy as np

######## KNN #########


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(x,y,query_point,k=5):
    vals=[]
    
    for i in range(x.shape[0]):
        d=dist(query_point,x[i])
        vals.append((d,y[i]))
        
    
    vals = sorted(vals)
    vals = vals[:k]
    vals=np.array(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    index=new_vals[1].argmax() #gives the index of the element having max count
    prediction=new_vals[0][index]
    return prediction

#######################


cap=cv.VideoCapture(0)

faceCascade=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
dataSet_path='./data/'

labels=[]
face_data=[]
names={} #dictionary for names of faces
class_id=0

for fx in os.listdir(dataSet_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        data_item=np.load(dataSet_path+fx)
        face_data.append(data_item)

        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)


face_dataSet=np.array(face_data)
faces_label=np.array(labels)

print(face_dataSet.shape)
print(faces_label.shape)

face_dataSet=np.concatenate(face_data,axis=0)
faces_label=np.concatenate(labels,axis=0)


# face_dataSet=face_dataSet.reshape((face_dataSet.shape[0]*face_dataSet.shape[1],-1))
# faces_label=faces_label.reshape((faces_label.shape[0]*faces_label.shape[1],))

print(face_dataSet.shape)
print(faces_label.shape)

#testing 

while True:
    ret,frame=cap.read()
    if ret == False:
        continue

    face_section=frame#because of error
    faces=faceCascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+offset+w]
        face_section=cv.resize(face_section,(200,200))

        out=knn(face_dataSet,faces_label,face_section.flatten())
        pred_name=names[int(out)]
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv.putText(frame,pred_name,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
    cv.imshow("Faces",frame)

    key_pressed=cv.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break

cap.release()
cv.destroyAllWindows()





