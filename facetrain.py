##Credits- https://youtu.be/PmZ29Vta7Vc
import os
from tkinter import Image
import PIL
import numpy as np,pandas as pd
from PIL import Image
import cv2
import pickle

face_cascade=cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

base_dir=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(base_dir,"images")

x_train=[]
y_label=[]
current_id=0
label_ids={}

for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" ","_").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id=current_id+1

            id_=label_ids[label]
            #print(label_ids)
            pil_image= Image.open(path).convert("L")
            size=(550,550)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array=np.array(final_image,"uint8")
            #print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.15,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_label.append(id_)

#print(y_label)
#print(x_train)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_label))
recognizer.save("trainner.yml")
