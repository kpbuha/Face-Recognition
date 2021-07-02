import cv2

cam =cv2.VideoCapture(0, cv2.CAP_DSHOW)#capture video through webcam
cam.set(3,640)#set Frramewidth
cam.set(4,480)#frameHeight

detector=cv2.CascadeClassifier("C:\\Users\\Karan Patel\\Desktop\\face recognition project\\haarcascade_frontalface_default.xml")
#this is effecticve object detection approach

face_id=input("Enter a Numeric use ID here: ")

print("Taking Sample, look at camera...")
count=0 #initializing sampling face count

while True:
    ret, img=cam.read()#read the frames
    converted_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#the function convert input img from one to
    faces=detector.detectMultiScale(converted_img,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#used to draw rectangle
        count+=1

        cv2.imwrite("samples\\face."+str(face_id)+'.'+str(count)+".jpg",converted_img[y:y+h,x:x+w])
        #to save images into dataset

        cv2.imshow('image',img)#used to display img in window

    k=cv2.waitKey(100) & 0xff #waits for a pressed key
    if k==27:
        break
    elif count >=10: #take 50 sample more sample more accouracy
        break

print("Samples taken now closing the program..")
cam.release()
cv2.destroyAllWindows()
