import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from tensorflow.keras.models import model_from_json

from cache.models import (
    cache_load_model,
    face_detect_model,
    LBPHRecognizer_model,
    left_right_eye_detect_model,
)
from constants.constants import (
    font,
    emotion_labels,
    blink_detection_path,
    emotion_model_path,
)
from sqlite.database import get_single_data



count = 0
score=0
thicc=2
rpred = 99
lpred = 99
w,h = 48,48


st.title("Real Time Emotion Detection")
wide_df = px.data.medals_wide()
placeholder = st.empty()                                            # creating a single-empty-element container
placeholder2 = st.empty()                                            # creating a single-empty-element container
placeholder3 = st.empty()                                            # creating a single-empty-element container
col1,col2 = st.columns(2)
run = st.checkbox('Open WebCam')
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)


# ==================================================
        # Sidebar Options (Checkbox)
# ==================================================
with st.sidebar:
    st.sidebar.subheader("Slider For Labels")
    label_size = st.sidebar.select_slider('Label Size',options=[1,2])

    st.sidebar.subheader('Chart')
    add_selectbox = st.sidebar.selectbox(
            "Do you like to see chart?",
            ("None", "Barplot")
        )

    
    st.subheader('Categories : ')
    facial_emotion_detection = st.sidebar.checkbox(
        'Facial Emotion Prediction'
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

# =====================================================
        # Different Model Initilization
# =====================================================


# Load Anti-Spoofing Model graph
json_file = open('../spoofing detection/models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
spoof_model = model_from_json(loaded_model_json)
# load antispoofing model weights 
spoof_model.load_weights('../spoofing detection/models/antispoofing_model.h5')
# spoof_model =load_model("E:/FINAL YEAR PROJECT/spoofing detection/models/spoof_antispoof_model.h5")


@st.cache(allow_output_mutation=True)
def model_face_func(model_path, type):
    if type == 'eigen':
        model_face = cv2.face.EigenFaceRecognizer_create()
        model_face.read(model_path)
        return model_face
    elif type == 'fisherman':
        model_face = cv2.face.FisherFaceRecognizer_create()
        model_face.read(model_path)
        return model_face
    elif type == 'lbph':
        model_face = LBPHRecognizer_model()
        model_face.read(model_path)
        return model_face
    else:
        model_face = cv2.face.EigenFaceRecognizer_create()
        model_face.read(model_path)
        return model_face




face_detection = face_detect_model()
leye,reye = left_right_eye_detect_model()
# age_model = cache_load_model(age_model_path)
# gender_model = cache_load_model(gender_model_path)
emotion_model = cache_load_model(emotion_model_path)
blink_detection_model = cache_load_model(blink_detection_path)

print("Model loaded from disk")
    

# ==================================================
        # WebCam Operations 
# ==================================================
vvvv= []
range1 = []
score1 = 0
score2 = 0
LEFT_EYE_FRAME_WINDOW = st.image([])
RIGHT_EYE_FRAME_WINDOW = st.image([])

if face_recognition:
                algorithm_selection1 = st.selectbox(
                    "Select Algorithm for Recognition",
                    ("LBPH Algorithm","Eigen Faces Algorithm","Fisher Faces Algorithm")
                )


# =======================================================
#  webcam videoCapture
# =======================================================
if run:
    while run:    
        ret, frame1 = cam.read()
        frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
        faces = face_detection.detectMultiScale(frame,1.1,5, minSize=(50,50))


        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = frame1[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_color,(48,48), interpolation=cv2.INTER_AREA)
            
            if facial_emotion_detection:
                roi_gray = cv2.cvtColor(roi_gray,cv2.COLOR_BGR2GRAY)
                roi_gray = roi_gray.reshape(1,48,48,1)
                roi_gray = roi_gray.astype("float") / 255.0
                emotion_prediction = prediction = emotion_model.predict(roi_gray)
                prediction_label = emotion_labels[np.argmax(prediction)]
                cv2.putText(frame,prediction_label,(x+w, y),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,255,0),1)

            
            if spoof_detection:
                resized_face = cv2.resize(frame,(160,160))
                resized_face = resized_face.astype("float") / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)
                preds = spoof_model.predict(resized_face)[0]
                if preds> 0.5:
                    label = 'spoof'
                else:
                    label = 'real'

                cv2.putText(frame,str(label),(x+w, y + 40),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,255,0),1)



            if face_recognition:
                roi_color = cv2.resize(roi_color,(224,224))
                roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)

                try:
                    if algorithm_selection1 == 'Fisher Faces Algorithm':
                        model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/fisher_face_classifier.yaml','fisherman')
                    elif algorithm_selection1 == 'Eigen Faces Algorithm':
                        model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/eigen_face_classifier.yaml','eigen')
                    else:
                        model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/classifier.yaml','lbph')
                except:
                    model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/classifier.yaml','lbph')


                prediction, conf = model_face.predict(roi_gray)
                if conf > 40:
                    print(prediction,conf)
                    data = get_single_data(prediction)
                    print(data)
                    cv2.putText(frame,'pred:'+str(data[1]),(x + w, y + h), font, label_size,(255,255,0),1,cv2.LINE_AA)
                    cv2.putText(frame,str(data[2]),(x, y), font, label_size,(255,255,0),1,cv2.LINE_AA)
                    cv2.putText(frame,str(data[3]),(x, y + h), font, label_size,(255,255,0),1,cv2.LINE_AA)
                else:
                    cv2.putText(frame,'Unknown',(x + w, y + h), font, label_size,(255,255,0),1,cv2.LINE_AA)
                
            
            if blink_detection:
                gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                left_eye = leye.detectMultiScale(gray)
                right_eye =  reye.detectMultiScale(gray)

                for (x,y,w,h) in right_eye:
                    r_eye=frame1[y:y+h,x:x+w]
                    count=count+1
                    r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                    r_eye = cv2.resize(r_eye,(24,24))
                    r_eye= r_eye/255
                    r_eye=  r_eye.reshape(24,24,-1)
                    r_eye = np.expand_dims(r_eye,axis=0)
                    rpred = blink_detection_model.predict(r_eye)
                    rpred = np.argmax(rpred)
                    print('rpred:::', rpred)

                    if(rpred==1):
                        lbl='Open'
                    if(rpred==0):
                        lbl='Closed'
                    RIGHT_EYE_FRAME_WINDOW.image(r_eye,caption="Right Eye")
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
                    vvvv.append(lpred[0][0])

                    lpred = np.argmax(lpred)
                    print('lpred:::', lpred)

                    if(lpred==1):
                        lbl='Open'   
                    if(lpred==0):
                        lbl='Closed'
                    LEFT_EYE_FRAME_WINDOW.image(l_eye,caption="Left Eye")
                    break

                if(rpred==0 and lpred==0):
                    score=score+1
                    score1=score1+1
                    cv2.putText(frame,"Eye: Closed",(10,400), font, 1,(255,255,0),1,cv2.LINE_AA)
                else:
                    score=score-1
                    score2=score2+1

                    cv2.putText(frame,"Eye: Open",(10, 400), font, 1,(255,255,0),1,cv2.LINE_AA)
                        
                if(score<0):
                    score=0   
                cv2.putText(frame,'Score: '+str(score),(10,380), font, 1,(255,255,0),1,cv2.LINE_AA)
                if score1 > 5 and score2 > 5:
                    cv2.putText(frame,'real',(20,20), font, 1,(255,255,0),1,cv2.LINE_AA)
                    score1 = score2 = 0
        
                # # Plot 
                with placeholder2.container():
                    if facial_emotion_detection and add_selectbox == 'Barplot':
                            x = emotion_labels
                            y = emotion_prediction[0] * 100
                            y = [int(i) for i in y]
                            fig = px.bar(
                                wide_df,
                                x=x,
                                y=y,
                                text=y,
                                color=x,
                                title="Facial Emotion Prediction",
                                height=400,
                                labels = {
                                    'x':'Facial Emotion class',
                                    'y': 'Facial Emotion Prediction in Perctange'
                                }
                            )
                            st.plotly_chart(fig)


                # if len(faces) < 2:
                with placeholder3.container():
                    if spoof_detection and add_selectbox == 'Barplot':
                        x = ['spoof','real']
                        y = [preds[0], 1 - preds[0]]
                        y = [int(i *  100) for i in y]

                        fig = px.bar(
                            wide_df,
                            x=x,
                            y=y,
                            text=y,
                            color=x,
                            title="Spoof Prediction",
                            height=400,
                            labels = {
                                'x':'Spoof Detection class',
                                'y': 'Spoof Prediction in Perctange'
                            }
                        )
                        st.plotly_chart(fig)
                
                with placeholder.container():
                    if add_selectbox == "Barplot" and blink_detection:
                        import datetime
                        import pandas as pd
                        if len(vvvv) != len(range1):
                            range1.append(datetime.datetime.now())
                        st.line_chart(pd.DataFrame(np.array(vvvv),np.array(range1)),height=300,width= 700,use_container_width=False)
                
        FRAME_WINDOW.image(frame)
else:
    cam.release()
    st.write("Stopped.")



