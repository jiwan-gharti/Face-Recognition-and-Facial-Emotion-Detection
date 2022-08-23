import cv2


@st.experimental_singleton 
def model_func():
    model = cv2.face.LBPHFaceRecognizer_create()
    return model
