import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import streamlit as st

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

# Start webcam
stframe = st.empty()
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Streamlit file uploader (optional test mode)
run = st.sidebar.checkbox("Run Virtual Mouse")

if run:
    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()

    st.info("Press **Stop** to end the session.")
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not detected!")
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

                x = int(index_finger.x * screen_w)
                y = int(index_finger.y * screen_h)

                # Move the mouse
                if enable_mouse:
                    pyautogui.moveTo(x, y)

                # Click gesture (distance between thumb and index)
                dist = np.hypot(thumb_finger.x - index_finger.x, thumb_finger.y - index_finger.y)
                if dist < 0.03 and enable_mouse:
                    pyautogui.click()
                    pyautogui.sleep(0.2)

        if show_feed:
            stframe.image(frame, channels="BGR", use_column_width=True)
        else:
            stframe.empty()
    cap.release()
else:
    st.info("👋 Turn on 'Run Virtual Mouse' from the sidebar to start using it!")

st.markdown("---")
st.caption("Developed by **Prachi Priyadarshini** | NIT Jamshedpur")
