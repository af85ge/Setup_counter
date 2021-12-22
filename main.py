import cv2
import mediapipe as mp
import numpy as np
import math

from numpy.lib.type_check import imag
counter_r = 0
counter_l = 0
r_true = True
l_true = True
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point.ravel()
    x1, y1 = point1.ravel()
    # print(x, x1)
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance
cap = cv2.VideoCapture("walking1.mp4")
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
    # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image)
    mp_drawing.draw_landmarks(
    image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # print(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value)
    if results.pose_landmarks:
      # print(results.pose_landmarks.landmark[31])
      coords =np.array([np.multiply([p.x, p.y, p.z], [width, height, width]).astype(int) for p in results.pose_landmarks.landmark])
      
      # [print(c) for c in coords]
      # print(coords[31][:2])
      right_z = coords[32][2]
      left_z = coords[31][2]

      cv2.putText(image, f'left {round(coords[31][2], 2)}',coords[31][:2],  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)
      cv2.putText(image,  f'Right {round(coords[32][2], 2)}', coords[32][:2], cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2, cv2.LINE_AA)
      cv2.line(image, pt1=coords[31][:2], pt2=coords[32][:2], color=(255,255,0),thickness=2, lineType= cv2.LINE_AA)
      cv2.circle(image, tuple(coords[32][:2]) , 6, (0,255,0), -1) # right foot index finger
      cv2.circle(image, tuple(coords[31][:2]) , 6, (0,255,0), -1) # left foot index finger
      # x, y  = coords[32].ravel()
      # x1, y1  = coords[32].ravel()
      # counting the steps 
      if right_z <-20 and l_true==True:
        counter_r +=1
        print('Right')
        r_true=True
        l_true =False
      if left_z <-20 and r_true==True:
        counter_l +=1
        print("left")
        r_true=False
        l_true = True
      
      cv2.putText(image, f"count_r: {counter_r} count_l: {counter_l} total: {counter_r+counter_r}", (30,40), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 2,cv2.LINE_AA)

      

      # print(x,y)
      distance_bt_feets =euclaideanDistance(coords[31][:2], coords[32][:2])

      # cv2.putText(image, f"Dsit: {round(distance_bt_feets,3)}", (30,40), cv2.FONT_HERSHEY_PLAIN, 1.6, (0,255,0), 2,cv2.LINE_AA)
      # print(distance_bt_feets)

      # if distance_bt_feets<70:
      #   cv2.putText(image, f"standing", (30,80), cv2.FONT_HERSHEY_PLAIN, 1.4, (0,255,255), 2,cv2.LINE_AA)
      # elif distance_bt_feets>100:
      #   cv2.putText(image, f"step", (30,80), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,0,255), 2,cv2.LINE_AA)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()