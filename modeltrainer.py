import cv2
import numpy as np
import os
from PIL import Image

path ="samples"

recognizer=cv2.face.LBPHFaceRecognizer_create()#local binary pattern histogram
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def Images_and_labels(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSample=[]
    ids=[]

    for imagePath in imagePaths:#to iterate particular image path

        gray_img=Image.open(imagePath).convert('L')
        img_arr=np.array(gray_img,'uint8')

        id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(img_arr)

        for(x,y,w,h) in faces:
            faceSample.append(img_arr[y:y+h,x:x+w])
            ids.append(id)
    return faceSample,ids

print("Training faces. It will take few seconds..")
faces,ids=Images_and_labels(path)
recognizer.train(faces,np.array(ids))
recognizer.write("trainer\\trainer.yml")#save the trained model as trainer.yml
print("Model trained, Now we can recognize your face.")