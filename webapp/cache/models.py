import cv2
import streamlit as st
from tensorflow.keras.models import load_model



@st.experimental_singleton 
def LBPHRecognizer_model():
    return cv2.face.LBPHFaceRecognizer_create()

@st.cache(allow_output_mutation=True)
def cache_load_model(model_name):
    return(load_model(model_name))


@st.cache(allow_output_mutation=True)
def face_detect_model():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache(allow_output_mutation=True)
def blink_detection_model():
    blink_detection = 'E:/FINAL YEAR PROJECT/blink detection/models/bestmodel.h5'
    return load_model(blink_detection)

@st.cache(allow_output_mutation=True)
def left_right_eye_detect_model():
    leye = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_righteye_2splits.xml')
    return leye,reye


