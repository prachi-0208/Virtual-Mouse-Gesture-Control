import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Hand Gesture Tracker (Deployable)", layout="wide")
st.title("🖐️ Deployable Hand Gesture Tracking")
st.markdown("This version runs successfully on Streamlit Cloud using `streamlit-webrtc` to access your camera.")
st.warning("Note: Virtual Mouse control (pynput/pyautogui) is **disabled** because it is not supported in cloud environments.")

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandGestureDetector(VideoTransformerBase):
    """A class to process video frames, detect hands, and draw landmarks."""
    def __init__(self):
        # Initialize MediaPipe Hands inside the transformer class
        self.hands = mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        # Convert frame from VideoFrame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr")
        
        # Flip the image for natural interaction (mirror effect)
        img = cv2.flip(img, 1)

        # Process the frame
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # BGR color for drawing
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

               
        return img


# --- STREAMLIT WEBRTC COMPONENT ---
st.sidebar.header("Camera Control")
st.sidebar.info("Click 'Start' to begin live hand tracking.")

webrtc_streamer(
    key="hand-gesture-tracker",
    video_transformer_factory=HandGestureDetector,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # Public STUN server for connection
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.caption("Deployment by Gemini | Core Logic by Prachi Priyadarshini, NIT Jamshedpur")
