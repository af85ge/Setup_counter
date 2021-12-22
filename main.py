import cv2
import mediapipe as mp
import numpy as np
import math
from numpy.lib.type_check import imag
mp_pose = mp.solutions.pose
# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point.ravel()
    x1, y1 = point1.ravel()
    # print(x, x1)
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    height, width = image.shape[:2]
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
      coords =np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.pose_landmarks.landmark])
      # [print(c) for c in coords]
      cv2.circle(image, tuple(coords[32]) , 6, (0,255,0), -1) # right foot index finger
      cv2.circle(image, tuple(coords[31]) , 6, (0,255,0), -1) # left foot index finger
      # x, y  = coords[32].ravel()
      # x1, y1  = coords[32].ravel()

      # print(x,y)
      distance_bt_feets =euclaideanDistance(coords[31], coords[32])
    
      cv2.putText(image, f"Dsit: {round(distance_bt_feets,3)}", (30,40), cv2.FONT_HERSHEY_PLAIN, 1.6, (0,255,0), 2,cv2.LINE_AA)
      print(distance_bt_feets)
      if distance_bt_feets<70:
        cv2.putText(image, f"standing", (30,80), cv2.FONT_HERSHEY_PLAIN, 1.4, (0,255,255), 2,cv2.LINE_AA)
      elif distance_bt_feets>100:
        cv2.putText(image, f"step", (30,80), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,0,255), 2,cv2.LINE_AA)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()