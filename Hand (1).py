%%writefile Hand.py
import cv2
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='/content/hand_landmarker.task')
options = vision.HandLandmarkerOptions( base_options=base_options, num_hands=1, min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

def recognise_gesture(landmarks):
    def finger_extended(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y

    def thumb_extended():
        return landmarks[4].x < landmarks[3].x

    thumb = thumb_extended()
    index = finger_extended(8, 6)
    middle = finger_extended(12, 10)
    ring = finger_extended(16, 14)
    pinky = finger_extended(20, 18)

    if all([thumb, index, middle, ring, pinky]):
        return "Open Palm"
    elif not any([thumb, index, middle, ring, pinky]):
        return "Fist"
    elif thumb and not any([index, middle, ring, pinky]):
        return "Thumbs Up"
    elif index and middle and not ring and not pinky:
        return "Peace Sign"
    elif pinky and not any([thumb, index, middle, ring]):
        return "Pinky"
    else:
        return "Unknown"

class videoProcessor:
  def recv(self, frame):
    frame = frame.to_ndarray(format="bgr24")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
      for hand_landmark in results.hand_landmarks:
        for landmark in hand_landmark:
          x = int(landmark.x * frame.shape[1])
          y = int(landmark.y * frame.shape[0])
          cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        #landmark = hand_landmark.landmark
        gesture = recognise_gesture(hand_landmark)
        cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")

st.title("Hand Gesture Recognition System")
st.write("Real Time Hand Detection and Recognition")

rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {'url': 'stun:global.stun.twilio.com:3478', 'urls': 'stun:global.stun.twilio.com:3478'},
{'credential': 'ZAEmMyLBBpCR4YjaLUI8R+4qKXyV+WdCl1XSfr7a4+I=', 'url': 'turn:global.turn.twilio.com:3478?transport=udp', 'urls': 'turn:global.turn.twilio.com:3478?transport=udp', 'username': 'b11fda7c87d0ae41c89793d1f7c2cf9c59eaad6c072571b084806a179d126e33'}, 
{'credential': 'ZAEmMyLBBpCR4YjaLUI8R+4qKXyV+WdCl1XSfr7a4+I=', 'url': 'turn:global.turn.twilio.com:3478?transport=tcp', 'urls': 'turn:global.turn.twilio.com:3478?transport=tcp', 'username': 'b11fda7c87d0ae41c89793d1f7c2cf9c59eaad6c072571b084806a179d126e33'},
{'credential': 'ZAEmMyLBBpCR4YjaLUI8R+4qKXyV+WdCl1XSfr7a4+I=', 'url': 'turn:global.turn.twilio.com:443?transport=tcp', 'urls': 'turn:global.turn.twilio.com:443?transport=tcp', 'username': 'b11fda7c87d0ae41c89793d1f7c2cf9c59eaad6c072571b084806a179d126e33'}
        ]
    }
)




webrtc_streamer(
    key="hand-recog",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=videoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)




webrtc_streamer(
    key="hand-recog",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=videoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
