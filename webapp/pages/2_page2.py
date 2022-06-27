import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 



count = 0
score=0
thicc=2
rpred = 99
lpred = 99
# creating a single-element container
placeholder = st.empty()

col1,col2 = st.columns(2)

with st.sidebar:
    st.sidebar.subheader("Slider For Labels")
    label_size = st.sidebar.select_slider('Label Size',options=[1,2])

    st.sidebar.subheader('Chart')
    add_selectbox = st.sidebar.selectbox(
            "Do you like to see chart?",
            ("None", "Barplot", "Countplot")
        )

    
    st.subheader('Categories : ')
    gray_image = st.sidebar.checkbox(
        'Gray Level Photo'
    )
    age_prediction_checkbox = st.sidebar.checkbox(
        "Age Range Prediction",
            # ("Email", "Home phone", "Mobile phone")
        )
    gender_prediction_checkbox = st.sidebar.checkbox(
        "Gender Classification"
    )
    facial_emotion_detection = st.sidebar.checkbox(
        'Facial Emotion Prediction'
    )
    race_classification = st.sidebar.checkbox(
        'Race Classification'
    )
    face_recognition = st.sidebar.checkbox(
        "Face Recognition"
    )
    spoof_detection = st.sidebar.checkbox(
        'Spoof Detection'
    )
    blink_detection = st.sidebar.checkbox(
        'Eye Blink Detection'
    )



model = load_model('E:/FINAL YEAR PROJECT/emotion detection/models/model_dropout.hdf5')

# Load Anti-Spoofing Model graph
json_file = open('E:/FINAL YEAR PROJECT/spoofing detection/models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
spoof_model = model_from_json(loaded_model_json)
# load antispoofing model weights 
spoof_model.load_weights('E:/FINAL YEAR PROJECT/spoofing detection/models/antispoofing_model.h5')
print("Model loaded from disk")


st.title("Real Time Emotion Detection")
run = st.checkbox('Open WebCam')

FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
w,h = 48,48
path_model = './Modelos/model_dropout.hdf5'
detect_frontal_face = '../../haar cascade files/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

blink_detection = 'E:/FINAL YEAR PROJECT/blink detection\models/bestmodel.h5'
blink_detection_model = load_model(blink_detection)
leye = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_righteye_2splits.xml')
lbl=['Close','Open']
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def barplot_function():
    fig = plt.figure(figsize=(9,7))
    print('------------------------------')
    # print(x,y)
    x = np.array(['real','spoof']),
    y = [1,0]
    bar = sns.barplot(x=x, y = y)


    st.pyplot(fig)
    

if run:
    while run:
        
        ret, frame = cam.read()
        height,width = frame.shape[:2] 
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_detection.detectMultiScale(frame,1.1,5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            # roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_color,(48,48), interpolation=cv2.INTER_AREA)
            roi_gray = cv2.cvtColor(roi_gray,cv2.COLOR_BGR2GRAY)
            roi_gray = roi_gray.reshape(1,48,48,1)
            roi_gray = roi_gray.astype("float") / 255.0
            prediction = model.predict(roi_gray)
            prediction_label = labels[np.argmax(prediction)]
            if gray_image:
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            cv2.putText(frame,prediction_label,(x+w, y),cv2.FONT_HERSHEY_COMPLEX,label_size,(0,255,0),2)


            if spoof_detection:
                resized_face = cv2.resize(frame,(160,160))
                resized_face = resized_face.astype("float") / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)
                preds = spoof_model.predict(resized_face)[0]
                if preds> 0.5:
                    label = 'spoof'
                else:
                    label = 'real'

                cv2.putText(frame,label,(x+w, y + 100),cv2.FONT_HERSHEY_COMPLEX,label_size,(0,255,0),2)

                
                with placeholder:
                    if add_selectbox == 'Barplot':
                        x = ['spoof','real']
                        y = [preds[0], 1 - preds[0]]
                        print(y)
                        fig = px.bar(x=x, y=y)
                        st.write(fig)

            if blink_detection:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                left_eye = leye.detectMultiScale(gray)
                right_eye =  reye.detectMultiScale(gray)

                for (x,y,w,h) in right_eye:
                    r_eye=frame[y:y+h,x:x+w]
                    count=count+1
                    r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                    r_eye = cv2.resize(r_eye,(24,24))
                    r_eye= r_eye/255
                    r_eye=  r_eye.reshape(24,24,-1)
                    r_eye = np.expand_dims(r_eye,axis=0)
                    rpred = blink_detection_model.predict(r_eye)
                    rpred = np.argmax(rpred)
                    if(rpred==1):
                        lbl='Open'
                    if(rpred==0):
                        lbl='Closed'
                    break

                for (x,y,w,h) in left_eye:
                    l_eye=frame[y:y+h,x:x+w]
                    count=count+1
                    l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                    l_eye = cv2.resize(l_eye,(24,24))
                    l_eye= l_eye/255
                    l_eye=l_eye.reshape(24,24,-1)
                    l_eye = np.expand_dims(l_eye,axis=0)
                    lpred = blink_detection_model.predict(l_eye)
                    lpred = np.argmax(lpred)

                    if(lpred==1):
                        lbl='Open'   
                    if(lpred==0):
                        lbl='Closed'
                    break

                if(rpred==0 and lpred==0):
                    score=score+1
                    cv2.putText(frame,"Eye: Closed",(10,height-20), font, 1,(255,255,0),1,cv2.LINE_AA)
                else:
                    score=score-1
                    cv2.putText(frame,"Eye: Open",(10,height-20), font, 1,(255,255,0),1,cv2.LINE_AA)
                
                        
                if(score<0):
                    score=0   
                cv2.putText(frame,'Score: '+str(score),(10,height-40), font, 1,(255,255,0),1,cv2.LINE_AA)

        FRAME_WINDOW.image(frame)
else:
    st.write("Stopped.")



