import cv2
import os
import random
import uuid
import time

from matplotlib import image


data_path = './outputs/webcam data'

cam = cv2.VideoCapture(0)

name = input("Enter your Name")


image_path = os.path.join(data_path, name)

if not os.path.exists(image_path):
    os.mkdir(os.path.join(data_path,name))
else:
    os.mkdir(os.path.join(data_path,f"{name} {uuid.uuid4()}"))


i = 0
while True:
    res, frame = cam.read()

    cv2.imwrite(f'{image_path}/image_{i}.png',frame)


    cv2.imshow('image',frame)

    if cv2.waitKey(1) == ord('q'):
        break
    
    i += 1







   
