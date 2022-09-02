

import cv2 

# age_labels = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
age_labels = ['1-2', '3-9', '10-20', '21-25','25-30', '31-45', '46-65']
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
gender_classes = ['man','woman']
lbl=['Close','Open']
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


# Path
blink_detection_path = 'E:/FINAL YEAR PROJECT/blink detection/models/bestmodel.h5'
emotion_model_path = 'E:/FINAL YEAR PROJECT/emotion detection/models/model_dropout.hdf5'
