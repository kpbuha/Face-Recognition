import cv2

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer\\trainer.yml")
cascadepath="haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascadepath)

font=cv2.FONT_HERSHEY_SIMPLEX

id=2

names=['','KARAN'] #leaves first empty because counter starts from zero

cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3,640)
cam.set(4,480)

minW=0.1*cam.get(3)#mmin window size to be recognized face
minH=0.1*cam.get(4)

while True:

    ret, img=cam.read()
    converted_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(
        converted_img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW),int(minH))
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        id,accuracy=recognizer.predict(converted_img[y:y+h,x:x+w])

        if(accuracy<100):
            id=names[id]
            accuracy=" {0}%".format(round(100 - accuracy))

        else:
            id="unknown"
            accuracy=" {0}%".format(round(100-accuracy))

        cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(accuracy),(x+5,y+h-5),font,1,(255,255,0),2)    
    
    cv2.imshow('camera',img)

    k=cv2.waitKey(10) & 0xff
    if k==27:
        break

print("thanks for using this program, have a good day")
cam.release()
cv2.destroyAllWindows()
