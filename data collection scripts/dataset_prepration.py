import cv2
import os

face_detection = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_frontalface_default.xml')


image_directory_path = 'E:/FINAL YEAR PROJECT/data collection scripts/shiva'
output_dir = 'E:/FINAL YEAR PROJECT/data collection scripts/shiva_face'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for i,image in enumerate(os.listdir(image_directory_path)):
    image_path = os.path.join(image_directory_path,image)
    image = cv2.imread(image_path)
    faces = face_detection.detectMultiScale(image,1.1,5)

    for (x,y,w,h) in faces:
        roi = image[y:y+h, x:x+w]
        cv2.imwrite(f'../data collection scripts/shiva_face/image_{i}.png',roi)

    if cv2.waitKey(1) == ord('q'):
        break
