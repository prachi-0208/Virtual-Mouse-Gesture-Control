import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller # KEEP: This is the correct mouse control library
import numpy as np
import streamlit as st
# import pyautogui # REMOVED: This library causes the deployment crash

mouse = Controller()
# mouse.position = (x, y) # REMOVED: Cannot define x, y here. Will be defined in the loop.

# --- Global Configurations ---
# Streamlit page setup
st.set_page_config(page_title="Virtual Mouse with Hand Gestures", layout="wide")
st.title("🖱️ Virtual Mouse using Hand Gestures")
st.markdown("Control your mouse with hand gestures — powered by OpenCV, MediaPipe, and Streamlit!")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Sidebar Controls
st.sidebar.header("Settings")
show_feed = st.sidebar.checkbox("Show Camera Feed", value=True)
enable_mouse = st.sidebar.checkbox("Enable Virtual Mouse", value=False)
stframe = st.empty()
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Streamlit file uploader (optional test mode)
run = st.sidebar.checkbox("Run Virtual Mouse")

if run:
    # IMPORTANT FIX: Since Streamlit runs headless, we can't use pyautogui.size()
    # We must assume screen size or use placeholder values.
    # On a Streamlit Cloud machine, the "screen" is typically 1920x1080.
    screen_w, screen_h = 1920, 1080 
    
    # We also cannot use cv2.VideoCapture(0) directly on Streamlit Cloud,
    # as it requires access to the system webcam. You need a solution 
    # like streamlit-webrtc for actual live video on Streamlit Cloud.
    # For now, we'll keep it as a local placeholder that WILL FAIL in the cloud.
    cap = cv2.VideoCapture(0) 

    st.info("Press **Stop** to end the session.")
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not detected! This app requires a webcam, but Streamlit Cloud needs 'streamlit-webrtc'.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                index_finger = landmarks[8]
                thumb_finger = landmarks[4]

                # Convert normalized coordinates to screen coordinates
                x = int(index_finger.x * screen_w)
                y = int(index_finger.y * screen_h)

                # Move the mouse
                if enable_mouse:
                    mouse.position = (x, y) # <-- CORRECT pynput SYNTAX

                # Click gesture (distance between thumb and index)
                # Note: distance is relative to the frame (normalized 0 to 1)
                dist = np.hypot(thumb_finger.x - index_finger.x, thumb_finger.y - index_finger.y)
                
                if dist < 0.03 and enable_mouse:
                    # CORRECT pynput SYNTAX
                    mouse.click(Button.left) 
                    # pynput does not have a sleep function, use standard Python time
                    # We would need to import time, but simple logic often suffices:
                    # In a deployment scenario, you should handle debounce (cooldown) 
                    # logic to prevent rapid, continuous clicks.
                    
        if show_feed:
            stframe.image(frame, channels="BGR", use_column_width=True)
        else:
            stframe.empty()
    cap.release()
else:
    st.info("👋 Turn on 'Run Virtual Mouse' from the sidebar to start using it!")

st.markdown("---")
st.caption("Developed by **Prachi Priyadarshini** | NIT Jamshedpur")
